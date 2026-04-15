import io
from rfdetr import RFDETRBase,RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES
from imutils import paths
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4"
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
    # model.train(
    #     dataset_dir=r'../LocalDataSetV12_COCO_RFDETR',
    #     epochs=200,
    #     batch_size=10,
    #     grad_accum_steps=2,
    #     lr=1e-4,
    #     output_dir=output_dir,
    #     class_names=['TA','LV'],
    #     use_ema=True,
    #     print_freq=1,
    #     # 新增的数据增强相关参数
    #     multi_scale=True,
    #     expanded_scales=True,
    #     do_random_resize_via_padding=True
    # )

    model.train(
        dataset_dir=r'../LocalDataSetV12_COCO_RFDETR',
        epochs=200,
        batch_size=10,
        grad_accum_steps=2,
        lr=1e-4,
        output_dir=output_dir,
        class_names=['TA', 'LV'],
        use_ema=True,
        print_freq=1,
        # 数据增强相关参数
        multi_scale=True,
        expanded_scales=True,
        do_random_resize_via_padding=True,
        # IoU感知的损失函数
        ia_bce_loss=True,                 # 使用IoU感知BCE损失
        cls_loss_coef=2,
        bbox_loss_coef=5,
        giou_loss_coef=2,
        focal_alpha=0.6,
        # # 针对长尾分布优化的损失函数配置
        # use_varifocal_loss=True,  # 使用Varifocal Loss替代标准Focal Loss
        # cls_loss_coef=5,  # 增加分类损失权重
        # focal_alpha=0.75,  # 调整alpha参数，给少数类更多权重
        # 学习率调度器配置
        lr_scheduler='cosine',
        lr_min_factor=0.001,  # lr_min / base_lr
        warmup_epochs=2,
        warmup_lr_init=1e-5,
        # 其他可能有用的参数
        # use_position_supervised_loss=True,  # 可选：位置监督损失
        # ia_bce_loss=True,               # 可选：IoU感知二值交叉熵损失
    )