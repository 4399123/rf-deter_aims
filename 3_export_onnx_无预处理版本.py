import torch
import torch.nn as nn
import onnx
import argparse
import os
import sys
from rfdetr import RFDETRNano
from onnxslim import slim



# --- 2. 定义全能包装器 (保持原逻辑不变) ---
class DeployModel(nn.Module):
    def __init__(self, core_model, max_detections=100):
        super().__init__()
        self.core_model = core_model
        self.max_detections = max_detections

        # 注册预处理常量 (ImageNet Mean/Std * 255)
        self.register_buffer('mean', torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))

    def forward(self, x):
        # --- 输入: [Batch, 3, H, W], uint8, BGR ---
        x = x.float()
        x = x[:, [2, 1, 0], :, :]  # BGR -> RGB
        x = (x - self.mean) / self.std

        # --- 推理 ---
        outputs = self.core_model(x)

        # --- 输出解包 ---
        if isinstance(outputs, dict):
            pred_boxes = outputs['pred_boxes']
            pred_logits = outputs['pred_logits']
        elif isinstance(outputs, (list, tuple)):
            pred_boxes = outputs[0]
            pred_logits = outputs[1]
        else:
            pred_boxes, pred_logits = outputs

        # --- 后处理 ---
        probs = pred_logits.sigmoid()
        max_scores, best_labels = probs.max(dim=-1)

        # 筛选 Top-K
        k = min(self.max_detections, pred_logits.shape[1])
        topk_scores, topk_indices = torch.topk(max_scores, k, dim=1, largest=True, sorted=True)

        # Gather
        indices_expanded_box = topk_indices.unsqueeze(-1).expand(-1, -1, 4)
        selected_boxes = torch.gather(pred_boxes, 1, indices_expanded_box)
        selected_labels = torch.gather(best_labels, 1, topk_indices)

        # Stack Output
        out_cx = selected_boxes[:, :, 0]
        out_cy = selected_boxes[:, :, 1]
        out_w = selected_boxes[:, :, 2]
        out_h = selected_boxes[:, :, 3]
        out_score = topk_scores
        out_cls = selected_labels.float()

        # [cx, cy, w, h, score, class_id]
        final_output = torch.stack([out_cx, out_cy, out_w, out_h, out_score, out_cls], dim=2)
        return final_output


def main(modelpath, onnx_raw_path, onnx_final_path, resolution, max_detections=100):
    print(f"Loading RFDETR from {modelpath}...")
    # 初始化模型
    rfdetr_wrapper = RFDETRNano(
        pretrain_weights=modelpath,
        resolution=resolution,
        device='cpu',
        num_queries=max_detections,
        num_select=max_detections
    )
    rfdetr_wrapper.optimize_for_inference()

    # 提取内核
    if hasattr(rfdetr_wrapper.model, 'model') and isinstance(rfdetr_wrapper.model.model, nn.Module):
        core_model = rfdetr_wrapper.model.model
        print("✅ Core model extracted (rfdetr.model.model)")
    elif isinstance(rfdetr_wrapper.model, nn.Module):
        core_model = rfdetr_wrapper.model
        print("✅ Core model extracted (rfdetr.model)")
    else:
        raise RuntimeError("❌ Failed to extract underlying PyTorch nn.Module")

    core_model.eval()

    # 包装模型
    deploy_model = DeployModel(core_model, max_detections=max_detections)
    deploy_model.eval()

    # 准备输入
    dummy_input = torch.zeros(1, 3, resolution, resolution, dtype=torch.uint8)

    # ----------------------------------------------------------------
    # Step 1: 导出原始 ONNX
    # ----------------------------------------------------------------
    print(f"\n[Step 1] Exporting raw ONNX to {onnx_raw_path}...")
    torch.onnx.export(
        deploy_model,
        dummy_input,
        onnx_raw_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=16,
        do_constant_folding=True
        # dynamic_axes 移除，保持 Static Shape 对 TensorRT 性能最好
    )

    # ----------------------------------------------------------------
    # Step 2: OnnxSlim (强力瘦身)
    # ----------------------------------------------------------------
    print(f"\n[Step 2] Optimizing with onnxslim...")
    try:
        # slim() 自动处理 Shape Inference 和复杂的死节点移除
        # 对于 DETR 类模型，这一步至关重要
        model_slimmed = slim(onnx_raw_path, model_check=True)
        print(" -> onnxslim complete.")
    except Exception as e:
        print(f"[Error] onnxslim failed: {e}")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Step 3: 直接保存优化后的模型
    # ----------------------------------------------------------------
    print(f"\n[Step 3] Saving optimized model...")
    onnx.save(model_slimmed, onnx_final_path)
    print(f" -> Model saved to {onnx_final_path}")

    # 清理临时文件
    # if os.path.exists(onnx_raw_path):
    #     os.remove(onnx_raw_path)

    # 打印结果
    final_size = os.path.getsize(onnx_final_path) / 1024 / 1024
    print(f"\n✅ SUCCESS!")
    print(f"Output Model: {onnx_final_path} ({final_size:.2f} MB)")
    print(f"Output Shape: [1, {max_detections}, 6]")
    print(f"Format: [cx, cy, w, h, score, class_id] (Normalized)")


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--weight-path', dest='weight_pth', type=str,
                       default='pt/v5/checkpoint_best_total.pth')

    parse.add_argument('--outpath', dest='out_pth', type=str,
                       default='./onnx/temp_raw.onnx')
    parse.add_argument('--ousmitpath', dest='outsmi_pth', type=str,
                       default='./onnx/best-smi.onnx')

    parse.add_argument('--resolution', type=int, default=384)
    parse.add_argument('--max-detections', type=int, default=150,
                       help='Maximum number of detections to output')
    args = parse.parse_args()

    # 确保目录存在
    os.makedirs(os.path.dirname(args.outsmi_pth), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_pth), exist_ok=True)

    main(args.weight_pth, args.out_pth, args.outsmi_pth, args.resolution, args.max_detections)