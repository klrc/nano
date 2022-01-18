import torch
from torch.utils.data import DataLoader
from nano.models.assigners.simota import (
    SimOTA,
)
from nano.models.model_zoo.nano_ghost import (
    GhostNano_3x4_m96,
    GhostNano_3x3_s64,
    GhostNano_3x3_s32,
)
from nano.datasets.class_utils import (
    coco_classes,
    voc_classes,
    c26_classes,
    create_class_mapping,
    _coco_to_c26,
    _coco_to_c3,
    _voc_to_c26,
)
from nano.datasets.coco_box2d import (
    MSCOCO,
    collate_fn,
    letterbox_collate_fn,
)
from nano.datasets.coco_box2d_transforms import (
    HSVTransform,
    SizeLimit,
    Affine,
    RandomScale,
    Albumentations,
    Mosaic4,
    ToTensor,
)
from nano.models.trainer import trainer, load_device
import wandb

if __name__ == "__main__":
    # ========================================================================

    for model_type, batch_size in (
        (GhostNano_3x4_m96, 128),
        # (GhostNano_3x3_s64, 128),
        # (GhostNano_3x3_s32, 128),
    ):
        # --------------------------------------------------
        try:

            class_names = c26_classes

            imgs_root = "../datasets/coco/images/train2017"
            annotations_root = "../datasets/coco/labels/train2017"
            coco_c26_mapping = create_class_mapping(coco_classes, c26_classes, _coco_to_c26)
            coco = MSCOCO(imgs_root=imgs_root, annotations_root=annotations_root, min_size=416, class_map=coco_c26_mapping)
            imgs_root = "../datasets/VOC/images/train2012"
            annotations_root = "../datasets/VOC/labels/train2012"
            voc_c26_mapping = create_class_mapping(voc_classes, c26_classes, _voc_to_c26)
            voc = MSCOCO(imgs_root=imgs_root, annotations_root=annotations_root, min_size=416, class_map=voc_c26_mapping)
            base = coco + voc

            # base = SizeLimit(base, 80000, targets=(0, 1, 2))
            base = RandomScale(base, p=1)
            base = Affine(base, p_flip=0.5, p_shear=0.2)
            base = HSVTransform(base, p=0.2)
            base = Albumentations(base, "random_blind")
            base = Mosaic4(base, img_size=448)
            trainset = ToTensor(base)
            imgs_root = "../datasets/coco/images/val2017"
            annotations_root = "../datasets/coco/labels/val2017"
            coco_c3_mapping = create_class_mapping(coco_classes, c26_classes, _coco_to_c3)
            base = MSCOCO(imgs_root=imgs_root, annotations_root=annotations_root, max_size=448, class_map=coco_c3_mapping)
            valset = ToTensor(base)

            train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=8, pin_memory=False, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(valset, batch_size=batch_size // 2, num_workers=8, pin_memory=False, collate_fn=letterbox_collate_fn)

            model = model_type(num_classes=len(class_names))
            device = load_device("cuda:1")
            criteria = SimOTA(len(class_names), True)

            logger = wandb.init(project="nano", dir="./runs", mode="offline")
            trainer.run(model, train_loader, val_loader, class_names, criteria, device, batch_size=batch_size, patience=10, epochs=150, wandb_logger=logger)
            del model, criteria, train_loader, val_loader, trainset, valset
        except Exception as e:
            print(e)
        finally:
            torch.cuda.empty_cache()
            logger.finish()
    # --------------------------------------------------
    # ========================================================================
