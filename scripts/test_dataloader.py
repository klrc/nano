import sys
import cv2
from loguru import logger

sys.path.append(".")

from nano.data.dataset import DatasetModule, CAVIARSeed, MSCOCOSeed  # noqa: E402
from nano.data.dataset_info import drive3_names, voc_to_drive3, coco_to_drive3  # noqa: E402
import nano.data.transforms as T  # noqa: E402
import nano.data.visualize as V  # noqa: E402


if __name__ == '__main__':
    try:
        logger.debug("Dataloader Test ------------------")

        target_resolution = (288, 512)
        factory = DatasetModule()
        # factory.add_seed(
        #     MSCOCOSeed(
        #         "../datasets/VOC/images/train2012",
        #         "../datasets/VOC/labels/train2012",
        #     ),
        #     T.IndexMapping(voc_to_drive3),
        #     T.HorizontalFlip(p=0.5),
        #     T.Resize(max_size=int(max(target_resolution) * 0.75)),  # 0.75x
        #     T.RandomScale(min_scale=0.6, max_scale=1),  # 0.8x
        #     T.RandomAffine(min_scale=0.875, max_scale=1.125, p=0.5),  # 1x
        #     T.HSVTransform(),
        #     T.AlbumentationsPreset(),
        #     T.Mosaic4(mosaic_size=int(max(target_resolution)), min_iou=0.45, p=1),
        #     T.ToTensor(),
        # )
        # factory.add_seed(
        #     MSCOCOSeed(
        #         "/Volumes/ASM236X/MSCOCO/train2017",
        #         "/Volumes/ASM236X/MSCOCO/labels/train2017",
        #     ),
        #     T.IndexMapping(coco_to_drive3),
        #     T.HorizontalFlip(p=0.5),
        #     T.Resize(max_size=int(max(target_resolution))),  # 1x
        #     T.RandomScale(min_scale=0.875, max_scale=1.125),  # 1x
        #     T.RandomAffine(min_scale=0.875, max_scale=1.125, p=0.5),  # 1x
        #     T.HSVTransform(),
        #     T.AlbumentationsPreset(),
        #     T.Mosaic4(mosaic_size=int(max(target_resolution)), min_iou=0.25, p=1),
        #     T.ToTensor(),
        # )
        # factory.add_seed(
        #     EmptySeed(
        #         "/Volumes/ASM236X/SKU110K_fixed/images",
        #         pick_rate=0.5,
        #     ),
        #     T.HorizontalFlip(p=0.5),
        #     T.Resize(max_size=int(max(target_resolution))),  # 1x
        #     T.ToTensor(),
        # )
        factory.add_seed(
            CAVIARSeed("/Volumes/ASM236X/CAVIAR", pick_rate=0.1),
            T.HorizontalFlip(p=0.5),
            T.ToTensor(),
        )
        factory = factory.as_dataloader(batch_size=16, num_workers=4, shuffle=True, collate_fn=T.letterbox_collate_fn)

        i = 0

        for images, target in factory:
            for i, image in enumerate(images):
                label = target[target[..., 0] == i][..., 1:]
                image = V.RenderLabels.functional(image, label, drive3_names)
                cv2.imwrite(f"test_image_{i}.png", image)
                logger.success(f"Load image at batch {i}")
            break

    except Exception as e:
        logger.error(e)
