from nano.data.dataset import detection_data_layer, Assembly
from nano.data.dataset_info import drive3_names, voc_to_drive3, coco_to_drive3
import nano.data.transforms as T
import nano.data.visualize as V
import cv2
from loguru import logger


def test_dataloader():
    try:
        logger.debug("Dataloader Test ------------------")

        target_resolution = (224, 416)
        dataset1 = detection_data_layer("../datasets/coco/images/train2017", "../datasets/coco/labels/train2017")
        dataset2 = detection_data_layer("../datasets/VOC/images/train2012", "../datasets/VOC/labels/train2012")
        factory = Assembly()
        factory.append_data(dataset1, dataset2)
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
        factory = factory.as_dataloader(batch_size=16, num_workers=4, collate_fn=T.letterbox_collate_fn)

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
