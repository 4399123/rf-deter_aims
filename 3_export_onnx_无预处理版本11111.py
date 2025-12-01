import torch
import torch.nn as nn
from rfdetr import RFDETRNano
import onnx
from onnxsim import simplify
import onnxoptimizer
import argparse


# --- 1. 定义全能包装器 (Pre/Post-Processing Wrapper) ---
class DeployModel(nn.Module):
    def __init__(self, core_model, max_detections=100):
        super().__init__()
        self.core_model = core_model
        self.max_detections = max_detections  # 存储可配置的最大检测数

        # 注册预处理常量 (ImageNet Mean/Std * 255)
        # 形状为 [1, 3, 1, 1] 以便广播
        self.register_buffer('mean', torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))

    def forward(self, x):
        # --- 输入: [Batch, 3, H, W], uint8, BGR ---

        # 1. 类型转换 (uint8 -> float)
        x = x.float()

        # 2. 通道转换 (BGR -> RGB)
        # OpenCV 读入是 BGR，模型训练用 RGB
        x = x[:, [2, 1, 0], :, :]

        # 3. 归一化 ( (x - mean) / std )
        x = (x - self.mean) / self.std

        # --- 推理 ---
        outputs = self.core_model(x)

        # --- 输出解包 ---
        # 兼容不同版本的返回格式
        if isinstance(outputs, dict):
            pred_boxes = outputs['pred_boxes']  # [Batch, 300, 4] (cx, cy, w, h) - 通常已经是归一化的
            pred_logits = outputs['pred_logits']  # [Batch, 300, num_classes]
        elif isinstance(outputs, (list, tuple)):
            pred_boxes = outputs[0]
            pred_logits = outputs[1]
        else:
            pred_boxes, pred_logits = outputs

        # --- 后处理 ---

        # 1. 计算所有类别的概率 [Batch, Queries, NumClasses]
        probs = pred_logits.sigmoid()

        # 2. 获取每个 Query 最大的分数和对应的类别索引
        # max_scores: [Batch, Queries], best_labels: [Batch, Queries]
        max_scores, best_labels = probs.max(dim=-1)

        # 3. 筛选 Top-K
        # 确保 k 不超过实际 query 数量
        k = min(self.max_detections, pred_logits.shape[1])

        # topk_scores: [Batch, K], topk_indices: [Batch, K]
        topk_scores, topk_indices = torch.topk(max_scores, k, dim=1, largest=True, sorted=True)

        # 4. Gather (收集) 对应的数据

        # 扩展 indices 以匹配 box 维度: [Batch, K] -> [Batch, K, 4]
        indices_expanded_box = topk_indices.unsqueeze(-1).expand(-1, -1, 4)

        # 收集对应的 Box [Batch, K, 4] (cx, cy, w, h)
        selected_boxes = torch.gather(pred_boxes, 1, indices_expanded_box)

        # 收集对应的 Label [Batch, K]
        selected_labels = torch.gather(best_labels, 1, topk_indices)

        # 5. 堆叠输出 [Batch, K, 6]
        # RF-DETR 的输出 pred_boxes 通常直接就是 [cx, cy, w, h] 且是归一化的(0-1)
        # 如果模型输出不是归一化的，这里需要除以 img_size，但标准 DETR 类模型都是归一化的
        out_cx = selected_boxes[:, :, 0]
        out_cy = selected_boxes[:, :, 1]
        out_w = selected_boxes[:, :, 2]
        out_h = selected_boxes[:, :, 3]
        out_score = topk_scores
        out_cls = selected_labels.float()  # 转换为 float 以便 stack

        # Format: [cx, cy, w, h, score, class_id]
        final_output = torch.stack([out_cx, out_cy, out_w, out_h, out_score, out_cls], dim=2)

        return final_output


def main(modelpath, onnxbest, onnxsmi, resolution, max_detections=100):
    rfdetr_wrapper = RFDETRNano(
        pretrain_weights=modelpath,
        resolution=resolution,
        device='cpu',
        num_queries=max_detections,  # 可配置的查询数量
        num_select=max_detections  # 可配置的选择数量
    )

    # 自动剥离外层封装，寻找真正的 nn.Module
    if hasattr(rfdetr_wrapper.model, 'model') and isinstance(rfdetr_wrapper.model.model, nn.Module):
        core_model = rfdetr_wrapper.model.model
        print("✅ 成功提取内核模型 (rfdetr.model.model)")
    elif isinstance(rfdetr_wrapper.model, nn.Module):
        core_model = rfdetr_wrapper.model
        print("✅ 成功提取内核模型 (rfdetr.model)")
    else:
        raise RuntimeError("❌ 无法找到底层的 PyTorch nn.Module，请检查库版本。")

    core_model.eval()

    # --- 3. 包装模型 ---
    deploy_model = DeployModel(core_model, max_detections=max_detections)
    deploy_model.eval()

    # --- 4. 准备虚拟输入 (Static Batch) ---
    # 形状: [1, 3, 672, 672], 类型: uint8
    dummy_input = torch.zeros(1, 3, resolution, resolution, dtype=torch.uint8)

    print("Exporting to ONNX...")
    torch.onnx.export(
        deploy_model,
        dummy_input,
        onnxbest,
        input_names=['input'],  # 输入节点名
        output_names=['output'],  # 输出节点名 (修改为单个 output)
        opset_version=16,  # 只能这个版本
        do_constant_folding=True,
        # dynamic_axes=...           # ❌ 已移除，强制使用静态 Batch
    )

    print('step 1 ok: Raw ONNX exported')
    model = onnx.load(onnxbest)

    print("Optimizing ONNX...")
    newmodel = onnxoptimizer.optimize(model)

    print("Simplifying ONNX...")
    model_simp, check = simplify(newmodel)
    assert check, "Simplified ONNX model could not be validated"

    onnx.save(model_simp, onnxsmi)
    print(f'step 2 ok: Optimized ONNX saved to {onnxsmi}')
    print(f'Output Shape: [Batch, {max_detections}, 6] -> [cx, cy, w, h, score, class_id]')


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--weight-path', dest='weight_pth', type=str,
                       default='pt/v4/checkpoint_best_total.pth')  # 训练好的模型路径
    parse.add_argument('--outpath', dest='out_pth', type=str,
                       default='./onnx/best.onnx')  # 中间产物onnx路径
    parse.add_argument('--ousmitpath', dest='outsmi_pth', type=str,
                       default='./onnx/best-smi.onnx')  # 最终产物onnx路径
    parse.add_argument('--resolution', type=int, default=384)  # 模型输入分辨率
    parse.add_argument('--max-detections', type=int, default=150,  # 最大检测数
                       help='Maximum number of detections to output')
    args = parse.parse_args()

    main(args.weight_pth, args.out_pth, args.outsmi_pth, args.resolution, args.max_detections)