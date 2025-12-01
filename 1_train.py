import io
from rfdetr import RFDETRBase,RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES
from imutils import paths
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
if __name__ == '__main__':
    model = RFDETRNano(pretrain_weights='weights/rf-detr-nano.pth', resolution=384,device='cuda')


    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = f'./runs/train_{timestamp}'

    # model.train(
    #     dataset_dir=r'C:\D\github_zl\LocalDataSetV12_COCO',
    #     epochs=1,
    #     batch_size=4,
    #     grad_accum_steps=1,
    #     lr=2e-4,
    #     output_dir=output_dir,
    # )
    model.train(
        dataset_dir=r'..\LocalDataSetV12_COCO_RFDETR',
        epochs=300,
        batch_size=16,
        grad_accum_steps=2,
        lr=1e-4,
        output_dir=output_dir,
        class_names=['TA','LV'],
        use_ema=False,
    )



