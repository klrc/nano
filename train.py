import torch
from torch.utils.data import DataLoader
from nano.models.assigners.simota import (
    SimOTA,
)
from nano.models.model_zoo.nano_ghost import (
    GhostNano_3x3_m96,
    GhostNano_3x3_s32,
    GhostNano_4x3_m96,
    GhostNano_4x4_l128,
    GhostNano_4x4_m96,
)
from nano.datasets.coco_box2d import MSCOCO, collate_fn, letterbox_collate_fn
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


if __name__ == "__main__":
    # ========================================================================

    for model_type, batch_size in (
        # (GhostNano_3x3_s32, 64),
        (GhostNano_3x3_m96, 16),
        (GhostNano_4x3_m96, 16),
        (GhostNano_4x4_m96, 16),
        (GhostNano_4x4_l128, 16),
    ):
        # --------------------------------------------------
        try:
            imgs_root = "/home/sh/Datasets/coco3/images/train"
            annotations_root = "/home/sh/Datasets/coco3/labels/train"
            base = MSCOCO(imgs_root=imgs_root, annotations_root=annotations_root, min_size=416)
            base = SizeLimit(base, 50000)
            base = RandomScale(base, p=0.2)
            base = Affine(base, p_flip=0.5, p_shear=0.2)
            base = HSVTransform(base, p=0.2)
            base = Albumentations(base, "random_blind")
            base = Mosaic4(base, img_size=448)
            trainset = ToTensor(base)
            imgs_root = "/home/sh/Datasets/coco3/images/val"
            annotations_root = "/home/sh/Datasets/coco3/labels/val"
            base = MSCOCO(imgs_root=imgs_root, annotations_root=annotations_root, max_size=416)
            valset = ToTensor(base)

            train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=8, pin_memory=False, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(valset, batch_size=batch_size // 2, num_workers=8, pin_memory=False, collate_fn=letterbox_collate_fn)

            model = model_type(num_classes=3)
            device = load_device("cuda")
            class_names = ["person", "bike", "car"]
            criteria = SimOTA(3, True)

            trainer.run(model, train_loader, val_loader, class_names, criteria, device, batch_size=batch_size)
            del model, criteria, train_loader, val_loader, trainset, valset
            torch.cuda.empty_cache()
        except Exception as e:
            torch.cuda.empty_cache()
            print(e)
    # --------------------------------------------------
    # ========================================================================
