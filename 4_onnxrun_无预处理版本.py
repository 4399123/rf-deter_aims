#encoding=gbk
import os.path

import onnx
import onnxruntime as ort
import numpy as np
import cv2
from tqdm import  tqdm
from imutils import paths
#路径配置
onnx_path=r'./onnx/best-smi.onnx'
imgspath=r'./images/baofeng'
# imgspath=r'./onnx/imgs'
w,h=384,384

if not os.path.exists('./results'):
    os.makedirs('./results')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

palette={0:(0,255,0),
    1:(0,0,255),
    2:(255,0,0),
    3:(255,255,0),
    4:(255,0,255),
    5:(171,130,255),
    6:(155,211,255),
    7:(0,255,255)}

label={0:'TuAn',
       1:'LV'}


imgpaths=list(paths.list_images(imgspath))

#onnx模型载入
model = onnx.load(onnx_path)
onnx.checker.check_model(model)
session = ort.InferenceSession(onnx_path,providers=['CPUExecutionProvider'])

for pic_path in tqdm(imgpaths):
    basename=os.path.basename(pic_path)
    name=basename.split('.')[0]
    img=cv2.imread(pic_path)
    H,W=img.shape[0],img.shape[1]
    h_ratio=H/h
    w_ratio=W/w
    imgbak=img.copy()
    img=cv2.resize(img,(w,h))
    img=np.array([np.transpose(img,(2,0,1))])


    #模型推理
    out = session.run(None, input_feed={'input': img})
    dets = out[0][0]  # [300, 4]  (通常是 cx, cy, w, h)
    scores_probs = out[1][0]  # [300, 91]

    # 后处理：解析检测结果
    detections = []
    for i in range(len(dets)):
        bbox = dets[i]
        probs = scores_probs[i]

        # 获取类别和置信度
        class_id = np.argmax(probs)
        confidence = probs[class_id]

        if confidence > 0.35:  # 置信度阈值
            # --- 修改点 1: 识别坐标格式 (cx, cy, w, h) ---
            # RF-DETR 原始输出通常是中心点坐标，都是 0-1 之间
            cx, cy, w_obj, h_obj = bbox

            # --- 修改点 2: 反归一化 (直接乘以原图 W, H) ---
            # 不要乘 ratio，要乘原图尺寸
            cx = cx * W
            cy = cy * H
            w_obj = w_obj * W
            h_obj = h_obj * H

            # --- 修改点 3: 转为左上角/右下角坐标 (xyxy) ---
            x1 = int(cx - w_obj / 2)
            y1 = int(cy - h_obj / 2)
            x2 = int(cx + w_obj / 2)
            y2 = int(cy + h_obj / 2)

            # 边界截断 (防止画图报错)
            x1 = max(0, min(x1, W))
            y1 = max(0, min(y1, H))
            x2 = max(0, min(x2, W))
            y2 = max(0, min(y2, H))

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'class_id': class_id,
                'confidence': confidence
            })

    # 可视化检测结果
    result_img = imgbak.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_id = det['class_id']
        confidence = det['confidence']

        # 绘制边界框
        color = palette.get(class_id % len(palette), (0, 255, 0))
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

        # 绘制标签和置信度
        label_text = f"{label.get(class_id, f'Class {class_id}')}: {confidence:.2f}"
        cv2.putText(result_img, label_text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 保存结果图像
    output_path = f'./results/{basename}'
    cv2.imwrite(output_path, result_img)
    print(f"Results saved to: {output_path}")

