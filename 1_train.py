import io
from rfdetr import RFDETRBase,RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES
from imutils import paths

if __name__ == '__main__':
    model = RFDETRNano(pretrain_weights='weights/rf-detr-nano.pth', resolution=640,device='cpu')


    model.train(
        dataset_dir=r'C:\D\github_zl\LocalDataSetV12_COCO',
        epochs=80,
        batch_size=4,
        grad_accum_steps=2,
        lr=2e-4,
        output_dir=r'./runs/train',  # 推荐固定写 runs/train，和 YOLO 一样
    )



