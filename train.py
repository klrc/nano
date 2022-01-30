from loguru import logger
from sklearn.utils import shuffle
import torch
from nano.data.dataset import detection_data_layer, Assembly
from nano.data.dataset_info import coco_to_drive3, voc_to_drive3, drive3_names
from nano.models.assigners.simota import SimOTA
from nano.models.model_zoo.nano_ghost import GhostNano_3x3_l128, GhostNano_3x3_m96, GhostNano_3x4_m96, GhostNano_3x4_s64, GhostNano_4x3_m96
from nano.training_core import Trainer, Validator, Controller
from nano.models.model_zoo import GhostNano_3x3_s64
import nano.data.transforms as T


if __name__ == "__main__":

    for model_template, target_resolution, batch_size in (
        (GhostNano_3x3_s64, (224, 416), 128),
        (GhostNano_3x4_s64, (224, 416), 128),
        (GhostNano_3x3_m96, (224, 416), 64),
        (GhostNano_3x4_m96, (224, 416), 64),
        (GhostNano_3x3_l128, (224, 416), 32),
    ):
        device = "cuda:0"

        trainset1 = detection_data_layer("../datasets/coco/images/train2017", "../datasets/coco/labels/train2017")
        trainset2 = detection_data_layer("../datasets/coco/images/val2017", "../datasets/coco/labels/val2017")
        trainset3 = detection_data_layer("../datasets/VOC/images/train2012", "../datasets/VOC/labels/train2012")
        factory = Assembly()
        factory.append_data(trainset1, trainset2, trainset3)
        factory.compose(
            T.ToNumpy(mark_source=True),
            T.IndexMapping(coco_to_drive3, pattern="/coco"),
            T.IndexMapping(voc_to_drive3, pattern="/VOC"),
            T.HorizontalFlip(p=0.5),
            T.Resize(max_size=max(target_resolution) + 64),
            T.RandomScale(min_scale=0.75, max_scale=1.25),
            T.RandomAffine(min_scale=0.875, max_scale=1.125, p=0.5),
            T.HSVTransform(),
            T.Mosaic4(mosaic_size=max(target_resolution) + 32, p=1),
            T.ToTensor(),
        )
        trainloader = factory.as_dataloader(batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=T.letterbox_collate_fn)
        valset = detection_data_layer("../datasets/VOC/images/val2012", "../datasets/VOC/labels/val2012")
        factory = Assembly()
        factory.append_data(valset)
        factory.compose(
            T.ToNumpy(mark_source=True),
            T.IndexMapping(coco_to_drive3, pattern="/coco"),
            T.IndexMapping(voc_to_drive3, pattern="/VOC"),
            T.Resize(max_size=max(target_resolution)),
            T.ToTensor(),
        )
        valloader = factory.as_dataloader(batch_size=batch_size // 2, num_workers=4, collate_fn=T.letterbox_collate_fn)

        model = model_template(len(drive3_names))
        criteria = SimOTA(len(drive3_names))

        trainer = Trainer(trainloader, model, criteria, device, lr0=0.001, batch_size=batch_size)
        validator = Validator(valloader, drive3_names, device)
        controller = Controller(trainer, validator)

        controller.run()
