
import pytorch_lightning as pl
from nano.datasets.object_detection import create_dataloader, check_dataset, colorstr
from nano.models.yolov5_cspdarknet_0_5x import Shell, Loss, yolov5s

if __name__ == '__main__':
    # load .yaml
    dataset_hyp = check_dataset('nano/configs/coco128.yaml')  # check
    augmentation_hyp = check_dataset('nano/configs/hyps/hyp.scratch.yaml')

    # Trainloader
    train_loader, dataset = create_dataloader(
        path=dataset_hyp['train'],     # dataset_path
        imgsz=416,                     # image_size
        batch_size=32,                 # batch_size
        stride=32,                     # grid_size
        single_cls=False,              #
        hyp=augmentation_hyp,          # augmentation_sets_yaml
        augment=True,                  # use_augmentation
        cache=False,                   # cache_image
        rect=False,                    #
        rank=-1,                       #
        workers=4,                     # num_workers
        quad=False,                    #
        prefix=colorstr('train: '),    # logger_display_title
    )
    # Pytorch-lightning shell    
    shell = Shell(yolov5s(), Loss())
    trainer = pl.Trainer()

    # run fit
    trainer.fit(Shell(), train_loader)