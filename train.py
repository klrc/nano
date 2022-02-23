from nano.data.dataset import DatasetModule, CAVIARSeed, IndoorSeed, MSCOCOSeed, PETS09Seed, SKU110KSeed, VIRATSeed  # noqa: E402
from nano.data.dataset_info import coco_to_drive3, voc_to_drive3, drive3_names
from nano.models.assigners.simota import SimOTA
from nano.models.model_zoo.nano_ghost import GhostNano_3x3_l128, GhostNano_3x3_m96
from nano.trainer.core import Trainer, Validator, Controller
import nano.data.transforms as T


if __name__ == "__main__":

    for model_template, target_resolution, batch_size in (
        (GhostNano_3x3_m96, (224, 416), 128),
        (GhostNano_3x3_l128, (224, 416), 64),
    ):
        device = "cuda:0"
        dataset_root = "../datasets"

        factory = DatasetModule()
        factory.add_seed(
            MSCOCOSeed(
                f"{dataset_root}/VOC/images/train2012",
                f"{dataset_root}/VOC/labels/train2012",
            ),
            T.IndexMapping(voc_to_drive3),
            T.HorizontalFlip(p=0.5),
            T.Resize(max_size=int(max(target_resolution) * 0.75)),  # 0.75x
            T.RandomScale(min_scale=0.6, max_scale=1),  # 0.8x
            T.RandomAffine(min_scale=0.875, max_scale=1.125, p=0.5),  # 1x
            T.HSVTransform(),
            T.AlbumentationsPreset(),
            T.Mosaic4(mosaic_size=int(max(target_resolution)), min_iou=0.45, p=1),
            T.ToTensor(),
        )
        factory.add_seed(
            MSCOCOSeed(
                f"{dataset_root}/MSCOCO/train2017",
                f"{dataset_root}/MSCOCO/labels/train2017",
            ),
            T.IndexMapping(coco_to_drive3),
            T.HorizontalFlip(p=0.5),
            T.Resize(max_size=int(max(target_resolution))),  # 1x
            T.RandomScale(min_scale=0.875, max_scale=1.125),  # 1x
            T.RandomAffine(min_scale=0.875, max_scale=1.125, p=0.5),  # 1x
            T.HSVTransform(),
            T.AlbumentationsPreset(),
            T.Mosaic4(mosaic_size=int(max(target_resolution)), min_iou=0.25, p=1),
            T.ToTensor(),
        )
#        factory.add_seed(
#            SKU110KSeed(f"{dataset_root}/SKU110K_fixed", pick_rate=0.1),
#            T.HorizontalFlip(p=0.5),
#            T.Resize(max_size=int(max(target_resolution))),  # 1x
#            T.ToTensor(),
#        )
        factory.add_seed(
            IndoorSeed(f"{dataset_root}/IndoorOD", pick_rate=0.2),
            T.HorizontalFlip(p=0.5),
            T.Resize(max_size=int(max(target_resolution))),  # 1x
            T.RandomAffine(min_scale=0.875, max_scale=1.125, p=0.5),  # 1x
            T.ToTensor(),
        )
        factory.add_seed(
            CAVIARSeed(f"{dataset_root}/CAVIAR", pick_rate=0.1),
            T.HorizontalFlip(p=0.5),
            T.Resize(max_size=int(max(target_resolution))),  # 1x
            T.RandomAffine(min_scale=0.875, max_scale=1.125, p=0.1),  # 1x
            T.HSVTransform(p=0.5),
            T.AlbumentationsPreset(),
            T.ToTensor(),
        )
        factory.add_seed(
            PETS09Seed(f"{dataset_root}/Crowd_PETS09", pick_rate=0.5),
            T.HorizontalFlip(p=0.5),
            T.Resize(max_size=int(max(target_resolution))),  # 1x
            T.RandomAffine(min_scale=0.875, max_scale=1.125, p=0.5),  # 1x
            T.HSVTransform(p=0.5),
            T.AlbumentationsPreset(),
            T.ToTensor(),
        )
        factory.add_seed(
            VIRATSeed(f"{dataset_root}/VIRAT", pick_rate=0.5),
            T.IndexMapping({1: 0, 2: 2, 3: 2, 5: 1}),
            T.HorizontalFlip(p=0.5),
            T.Resize(max_size=int(max(target_resolution))),
            T.RandomScale(min_scale=1, max_scale=1.5),
            T.RandomAffine(min_scale=0.875, max_scale=1.125, p=0.5),  # 1x
            T.HSVTransform(p=0.5),
            T.AlbumentationsPreset(),
            T.ToTensor(),
        )
        trainloader = factory.as_dataloader(batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=T.letterbox_collate_fn)
        factory = DatasetModule()
        factory.add_seed(
            MSCOCOSeed(
                f"{dataset_root}/VOC/images/val2012",
                f"{dataset_root}/VOC/labels/val2012",
            ),
            T.IndexMapping(voc_to_drive3),
            T.Resize(max_size=int(max(target_resolution))),
            T.ToTensor(),
        )
        valloader = factory.as_dataloader(batch_size=batch_size // 2, num_workers=4, collate_fn=T.letterbox_collate_fn)

        model = model_template(len(drive3_names))
        criteria = SimOTA(len(drive3_names))

        trainer = Trainer(trainloader, model, criteria, device, lr0=0.001, batch_size=batch_size)
        validator = Validator(valloader, drive3_names, device)
        controller = Controller(trainer, validator)

        controller.run()
