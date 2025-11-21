import io
from rfdetr import RFDETRBase,RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES
from imutils import paths

if __name__ == '__main__':
    model = RFDETRNano(pretrain_weights='pt/v1/checkpoint_best_regular.pth', resolution=640,device='cpu')
    model.optimize_for_inference()


    model.export(format="onnx",opset=16,simplify=True)



