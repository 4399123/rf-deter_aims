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
            pred_boxes = outputs['pred_boxes']  # [Batch, 300, 4] (cx, cy, w, h)
            pred_logits = outputs['pred_logits']  # [Batch, 300, 91]
        elif isinstance(outputs, (list, tuple)):
            pred_boxes = outputs[0]
            pred_logits = outputs[1]
        else:
            pred_boxes, pred_logits = outputs

        # --- 后处理 ---
        # 4. 只保留 top-k 预测 (确保不超过模型设置的num_queries)
        max_output = min(self.max_detections, pred_logits.shape[1])  # 使用可配置的最大检测数
        topk_values, topk_indices = torch.topk(pred_logits.sigmoid().max(-1)[0], k=max_output, dim=1)
        pred_boxes = torch.gather(pred_boxes, 1, topk_indices.unsqueeze(-1).expand(-1, -1, 4))
        pred_logits = torch.gather(pred_logits, 1, topk_indices.unsqueeze(-1).expand(-1, -1, pred_logits.shape[-1]))
        pred_scores = torch.sigmoid(pred_logits)

        # 返回: 归一化的坐标框, 0-1的置信度
        return pred_boxes, pred_scores

def main(modelpath,onnxbest,onnxsmi,resolution,max_detections=100):

    rfdetr_wrapper = RFDETRNano(
        pretrain_weights=modelpath,
        resolution=resolution,
        device='cpu',
        num_queries=max_detections,  # 可配置的查询数量
        num_select=max_detections    # 可配置的选择数量
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


    torch.onnx.export(
        deploy_model,
        dummy_input,
        onnxbest,
        input_names=['input'],  # 输入节点名
        output_names=['boxes', 'scores'],  # 输出节点名
        opset_version=16,  # 只能这个版本
        do_constant_folding=True,
        # dynamic_axes=...              # ❌ 已移除，强制使用静态 Batch
    )


    print('step 1 ok')
    model = onnx.load(onnxbest)

    newmodel = onnxoptimizer.optimize(model)

    model_simp, check = simplify(newmodel)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnxsmi)
    print('step 2 ok')

if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument('--weight-path', dest='weight_pth', type=str,
                       default='pt/v1/checkpoint_best_regular.pth')          #训练好的模型路径
    parse.add_argument('--outpath', dest='out_pth', type=str,
                       default='./onnx/best.onnx')                     #中间产物onnx路径
    parse.add_argument('--ousmitpath', dest='outsmi_pth', type=str,
                       default='./onnx/best-smi.onnx')                 #最终产物onnx路径
    parse.add_argument('--resolution', type=int,default=640)      #模型输入分辨率
    parse.add_argument('--max-detections', type=int, default=150,   #最大检测数
                      help='Maximum number of detections to output')
    args = parse.parse_args()


    main(args.weight_pth,args.out_pth,args.outsmi_pth,args.resolution,args.max_detections)