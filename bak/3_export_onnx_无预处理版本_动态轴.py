import torch
import torch.nn as nn
from rfdetr import RFDETRNano
import onnx
from onnx import shape_inference
import argparse


# --- 1. JIT Script (关键修改：显式 reshape) ---
@torch.jit.script
def post_process_script(pred_boxes, pred_logits, max_detections: int, num_classes: int):
    # pred_boxes: [Batch, 300, 4]
    # pred_logits: [Batch, 300, 91]

    batch_size = pred_logits.size(0)

    # Sigmoid
    probs = pred_logits.sigmoid()

    # TopK
    topk_values, topk_indices = torch.topk(probs.max(-1)[0], k=max_detections, dim=1)

    # 构造 Gather 索引
    # [Batch, K] -> [Batch, K, 1] -> [Batch, K, 4]
    indices_boxes = topk_indices.unsqueeze(-1).expand(batch_size, max_detections, 4)

    # [Batch, K] -> [Batch, K, C]
    indices_logits = topk_indices.unsqueeze(-1).expand(batch_size, max_detections, num_classes)

    # Gather
    new_boxes = torch.gather(pred_boxes, 1, indices_boxes)
    new_probs = torch.gather(probs, 1, indices_logits)

    # --- 关键修改点 ---
    # 强制 Reshape:
    # -1 表示 Batch 维度继续保持动态
    # max_detections, 4, num_classes 是我们传入的整数，ONNX 会把它们识别为固定常数
    new_boxes = new_boxes.view(-1, max_detections, 4)
    new_probs = new_probs.view(-1, max_detections, num_classes)

    return new_boxes, new_probs


# --- 2. 包装器 ---
class DeployModel(nn.Module):
    def __init__(self, core_model, max_detections=100, num_classes=91):
        super().__init__()
        self.core_model = core_model
        self.max_detections = max_detections
        self.num_classes = num_classes  # 记录类别数

        # 注册预处理常量
        self.register_buffer('mean', torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))

    def forward(self, x):
        # 预处理
        x = x.float()
        x = x[:, [2, 1, 0], :, :]
        x = (x - self.mean) / self.std

        outputs = self.core_model(x)

        if isinstance(outputs, dict):
            pred_boxes = outputs['pred_boxes']
            pred_logits = outputs['pred_logits']
        elif isinstance(outputs, (list, tuple)):
            pred_boxes = outputs[0]
            pred_logits = outputs[1]
        else:
            pred_boxes, pred_logits = outputs

        # 这里的 91 (或 80) 通常是固定的，我们直接传进去
        # 注意：RF-DETR COCO 默认是 91 (含背景位) 或 80，取决于具体权重
        # 我们这里通过 pred_logits.shape[-1] 获取真实值传给 JIT
        real_num_classes = pred_logits.shape[-1]

        return post_process_script(pred_boxes, pred_logits, self.max_detections, real_num_classes)


def main(modelpath, onnxbest, resolution, max_detections=100):
    # 1. 初始化
    rfdetr_wrapper = RFDETRNano(
        pretrain_weights=modelpath,
        resolution=resolution,
        device='cpu',
        num_queries=max_detections,
        num_select=max_detections
    )

    if hasattr(rfdetr_wrapper.model, 'model') and isinstance(rfdetr_wrapper.model.model, nn.Module):
        core_model = rfdetr_wrapper.model.model
    elif isinstance(rfdetr_wrapper.model, nn.Module):
        core_model = rfdetr_wrapper.model
    else:
        raise RuntimeError("❌ 无法找到底层的 PyTorch nn.Module")

    core_model.eval()

    # 2. 包装
    deploy_model = DeployModel(core_model, max_detections=max_detections)
    deploy_model.eval()

    # 3. 导出
    dummy_input = torch.zeros(1, 3, resolution, resolution, dtype=torch.uint8)

    dynamic_axes_config = {
        'input': {0: 'batch'},
        'boxes': {0: 'batch'},
        'scores': {0: 'batch'}
    }

    print(f"正在导出 ONNX (Res={resolution}, TopK={max_detections})...")

    # 导出
    torch.onnx.export(
        deploy_model,
        dummy_input,
        onnxbest,
        input_names=['input'],
        output_names=['boxes', 'scores'],
        opset_version=16,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes_config
    )

    print(f"✅ 导出完成，正在进行形状推断(Shape Inference)...")

    # 4. 关键后处理：形状推断
    # 这一步会计算出所有能确定的形状，把那些 GatherElements... 变成真正的数字
    try:
        model = onnx.load(onnxbest)
        # 推断形状
        model = shape_inference.infer_shapes(model)
        onnx.save(model, onnxbest)

        print(f"🎉 完美处理完成: {onnxbest}")
        print("   - 输入: [batch, 3, H, W]")
        # 打印最终形状验证
        out0_shape = [d.dim_param if d.dim_param else d.dim_value for d in
                      model.graph.output[0].type.tensor_type.shape.dim]
        out1_shape = [d.dim_param if d.dim_param else d.dim_value for d in
                      model.graph.output[1].type.tensor_type.shape.dim]

        print(f"   - boxes:  {out0_shape}  (期望: ['batch', {max_detections}, 4])")
        print(f"   - scores: {out1_shape} (期望: ['batch', {max_detections}, 91])")

    except Exception as e:
        print(f"⚠️ 形状推断警告: {e}")


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--weight-path', dest='weight_pth', type=str, default='pt/v1/checkpoint_best_regular.pth')
    parse.add_argument('--outpath', dest='out_pth', type=str, default='./onnx/best-smi.onnx')
    parse.add_argument('--resolution', type=int, default=384)
    parse.add_argument('--max-detections', type=int, default=150)
    args = parse.parse_args()

    main(args.weight_pth, args.out_pth, args.resolution, args.max_detections)