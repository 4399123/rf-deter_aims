# encoding=gbk
import os.path
import onnx
import onnxruntime as ort
import numpy as np
import cv2
from tqdm import tqdm
from imutils import paths

# 路径配置
onnx_path = r'./onnx/best-smi.onnx'
imgspath = r'./images/baofeng'
results_path = './results_batch'
w, h = 640, 640
BATCH_SIZE = 2  # 指定 Batch Size 为 2

if not os.path.exists(results_path):
    os.makedirs(results_path)

palette = {
    0: (0, 255, 0),
    1: (0, 0, 255),
    2: (255, 0, 0),
    3: (255, 255, 0),
    4: (255, 0, 255),
    5: (171, 130, 255),
    6: (155, 211, 255),
    7: (0, 255, 255)
}

label = {0: 'TuAn', 1: 'LV'}

# 获取所有图片路径
imgpaths = list(paths.list_images(imgspath))

# onnx模型载入
model = onnx.load(onnx_path)
onnx.checker.check_model(model)

# 启用 CUDA 或 CPU
# 如果你的 onnxruntime-gpu 环境配置好了，建议把 'CUDAExecutionProvider' 放在前面
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession(onnx_path, providers=providers)

print(f"Model Input Name: {session.get_inputs()[0].name}")
print(f"Model Input Shape: {session.get_inputs()[0].shape}")
print(f"Testing with Batch Size: {BATCH_SIZE}")

# --- 批处理循环 ---
# 每次步进 BATCH_SIZE 个单位
for i in range(0, len(imgpaths), BATCH_SIZE):
    # 1. 获取当前 Batch 的图片路径列表
    batch_paths = imgpaths[i: i + BATCH_SIZE]

    # 如果剩下的图片不足一个 Batch（例如最后只剩1张），也可以运行，
    # 但为了严格测试你的 "batch=2" 需求，你可以决定是否跳过，这里默认兼容动态处理。
    current_batch_size = len(batch_paths)

    batch_images = []  # 存放预处理后的图像数据
    original_metas = []  # 存放原图信息（原图数据，原宽，原高，文件名），用于后处理

    # 2. 预处理 Batch 中的每一张图
    for pic_path in batch_paths:
        basename = os.path.basename(pic_path)
        img_raw = cv2.imread(pic_path)

        # 记录原图信息，供后处理画图使用
        H_orig, W_orig = img_raw.shape[0], img_raw.shape[1]
        original_metas.append({
            "img": img_raw.copy(),
            "H": H_orig,
            "W": W_orig,
            "name": basename
        })

        # Resize 和 Transpose
        img_resized = cv2.resize(img_raw, (w, h))
        # HWC -> CHW
        img_transposed = np.transpose(img_resized, (2, 0, 1))
        batch_images.append(img_transposed)

    # 3. 堆叠成 Batch Tensor
    # 结果形状: (batch_size, 3, 640, 640)
    input_tensor = np.array(batch_images)

    print(f"\nProcessing Batch: {[m['name'] for m in original_metas]}")
    print(f"Input Tensor Shape: {input_tensor.shape}")

    # 4. 模型推理 (一次推多张)
    # out[0]: boxes, out[1]: scores
    # 假设 out[0] shape 为 (batch_size, 300, 4)
    out = session.run(None, input_feed={'input': input_tensor})

    batch_dets = out[0]  # [batch_size, 300, 4]
    batch_scores_probs = out[1]  # [batch_size, 300, 91]

    # 5. 后处理：解析 Batch 中的每一个结果
    for b_i in range(current_batch_size):
        # 取出当前这第 b_i 张图的数据
        dets = batch_dets[b_i]
        scores_probs = batch_scores_probs[b_i]

        # 取出对应的原图元数据
        meta = original_metas[b_i]
        img_vis = meta['img']
        W_real = meta['W']
        H_real = meta['H']
        filename = meta['name']

        detections = []
        for k in range(len(dets)):
            bbox = dets[k]
            probs = scores_probs[k]

            class_id = np.argmax(probs)
            confidence = probs[class_id]

            if confidence > 0.35:
                # RF-DETR 输出通常是 (cx, cy, w, h)，范围 0-1
                cx, cy, w_obj, h_obj = bbox

                # --- 关键修改：使用各自原图的 W_real, H_real 进行反归一化 ---
                cx = cx * W_real
                cy = cy * H_real
                w_obj = w_obj * W_real
                h_obj = h_obj * H_real

                x1 = int(cx - w_obj / 2)
                y1 = int(cy - h_obj / 2)
                x2 = int(cx + w_obj / 2)
                y2 = int(cy + h_obj / 2)

                # 边界截断
                x1 = max(0, min(x1, W_real))
                y1 = max(0, min(y1, H_real))
                x2 = max(0, min(x2, W_real))
                y2 = max(0, min(y2, H_real))

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'class_id': class_id,
                    'confidence': confidence
                })

        # 可视化并保存
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = det['class_id']
            confidence = det['confidence']

            color = palette.get(class_id % len(palette), (0, 255, 0))
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)

            label_text = f"{label.get(class_id, f'Class {class_id}')}: {confidence:.2f}"
            cv2.putText(img_vis, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        output_file = os.path.join(results_path, filename)
        cv2.imwrite(output_file, img_vis)
        print(f"  -> Saved: {output_file}")

print("\nBatch inference test completed.")