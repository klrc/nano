
from nano.models.yolov5_shufflenet_1_5x import Shell, Loss, yolov5_shufflenet_1_5x
from nano.datasets.object_detection import create_dataloader, check_dataset, colorstr, check_file_and_load, select_device
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

if __name__ == '__main__':
    # load device &.yaml
    batch_size = 16
    device = select_device(batch_size=batch_size)
    dataset_hyp = check_dataset('nano/configs/coco-s.yaml')  # check
    hyp = check_file_and_load('nano/configs/hyps/hyp.scratch.yaml')

    # Trainloader
    train_loader, dataset = create_dataloader(
        path=dataset_hyp['train'],     # dataset_path
        imgsz=416,                     # image_size
        batch_size=batch_size,         # batch_size
        stride=32,                     # grid_size
        single_cls=False,              #
        hyp=hyp,                       # augmentation_sets_yaml
        augment=True,                  # use_augmentation
        cache=False,                   # cache_image
        rect=False,                    #
        rank=-1,                       #
        workers=4,                     # num_workers
        quad=False,                    #
        prefix=colorstr('train: '),    # logger_display_title
    )
    # Pytorch-lightning shell
    nc = 6
    anchors = ([10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326])
    wandb_logger = WandbLogger(name='yolov5_shufflenet_1_5x', project='nano-coco-s')
    shell = Shell(
        yolov5_shufflenet_1_5x(num_classes=nc, anchors=anchors),
        Loss(hyp, nc=nc, anchors=anchors), device)
    trainer = pl.Trainer(gpus=1, logger=wandb_logger)

    # run fit
    trainer.fit(shell, train_loader)
