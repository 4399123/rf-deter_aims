import io
import requests
import matplotlib
import  os
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase,RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES
from imutils import paths

model = RFDETRNano(pretrain_weights='weights/rf-detr-nano.pth', resolution=672)
model.optimize_for_inference()

input_path=r'./images'
output_path=r'./results'

if not os.path.exists(output_path):
    os.mkdir(output_path)

imagespaths=list(paths.list_images(input_path))

for img_path in imagespaths:
    basename = os.path.basename(img_path)
    image=Image.open(img_path)
    detections=model.predict(image,threshold=0.5)
    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]
    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

    annotated_image.save(os.path.join(output_path,basename),quality=100)





