import sys
import cv2
from loguru import logger

sys.path.append(".")


if __name__ == "__main__":
    try:
        logger.debug("Dataloader Test ------------------")
        from nano.data.dataset import person_vehicle_detection_preset, preson_vehicle_detection_preset_mscoco_test
        import nano.data.transforms as T
        import nano.data.visualize as V

        class_names = "person|bike|motorcycle|car|bus|truck|OOD".split("|")
        # dataset = preson_vehicle_detection_preset_mscoco_test((256, 448), "person|bike|motorcycle|car|bus|truck|OOD", "/Volumes/ASM236X")
        dataset = person_vehicle_detection_preset((256, 448), class_names, "/Volumes/ASM236X")
        dataloader = dataset.as_dataloader(batch_size=16, num_workers=4, shuffle=True, collate_fn=T.letterbox_collate_fn)

        for images, target in dataloader:
            for i, image in enumerate(images):
                label = target[target[..., 0] == i][..., 1:]
                image = V.RenderLabels.functional(image, label, class_names)
                cv2.imwrite(f"test_image_{i}.png", image)
                logger.success(f"Load image at batch {i}")
            break

    except Exception as e:
        logger.error(e)
        raise e
