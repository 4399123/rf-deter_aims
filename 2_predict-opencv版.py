import  cv2
import  os
import numpy as np
from PIL import Image
from rfdetr import RFDETRSmall,RFDETRNano,RFDETRMedium
from rfdetr.util.coco_classes import COCO_CLASSES
from class_label.baofeng_class import BF_CLASSES
from imutils import paths

resolution=640
model = RFDETRNano(pretrain_weights='pt/v3/checkpoint_best_regular.pth', resolution=resolution)
model.optimize_for_inference()

# input_path=r'./images/baofeng'
input_path=r'C:\D\github_zl\LocalDataSetV12_COCO\train'
output_path=r'./results'

if not os.path.exists(output_path):
    os.mkdir(output_path)


# 定义颜色生成函数（为每个类别分配固定颜色）
def generate_colors(num_classes):
    """为每个类别生成不同的颜色 (BGR格式)"""
    np.random.seed(42)  # 保证每次运行颜色一致
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    return colors

CLASS_COLORS = generate_colors(256)


imagespaths=list(paths.list_images(input_path))

for img_path in imagespaths:
    basename = os.path.basename(img_path)
    image=cv2.imread(img_path)
    imgage = cv2.resize(image, (resolution, resolution))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections=model.predict(image,threshold=0.1)

    # 获取检测信息
    boxes = detections.xyxy  # 边界框坐标 [x1, y1, x2, y2]
    class_ids = detections.class_id
    confidences = detections.confidence

    # 遍历每个检测结果并绘制
    for box, class_id, confidence in zip(boxes, class_ids, confidences):
        # 获取坐标（转换为整数）
        x1, y1, x2, y2 = map(int, box)

        # 获取类别名称和颜色
        class_name = BF_CLASSES[class_id]
        color = CLASS_COLORS[class_id].tolist()  # 获取该类别的颜色

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)

        # 准备标签文本
        label = f"{class_name} {confidence:.2f}"

        # 计算标签文本的尺寸
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1
        )

        # 绘制标签背景矩形
        cv2.rectangle(image,(x1, y1 - text_height - 10),(x1 + text_width, y1),color,thickness=-1)# -1 表示填充矩形

        # 绘制标签文本
        cv2.putText(image,label,(x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(255, 255, 255), thickness=1)
    cv2.imwrite(os.path.join(output_path,'opencv'+basename),image[:,:,::-1])







