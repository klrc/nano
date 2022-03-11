import sys
import cv2
from loguru import logger


sys.path.append(".")


if __name__ == "__main__":
    try:
        logger.debug("Image Detection Test ------------------")
        from nano.data.dataset import person_vehicle_detection_preset, preson_vehicle_detection_preset_mscoco_test
        from nano.models.multiplex.box2d import non_max_suppression
        from nano.models.model_zoo import GhostNano_3x3_m96
        import nano.data.transforms as T
        import nano.data.visualize as V

        # dataset = preson_vehicle_detection_preset_mscoco_test((256, 448), "person|bike|motorcycle|car|bus|truck|OOD", "/Volumes/ASM236X")
        class_names = "person|bike|motorcycle|car|bus|truck|OOD".split("|")
        dataset = person_vehicle_detection_preset((256, 448), class_names, "/Volumes/ASM236X")
        dataloader = dataset.as_dataloader(batch_size=16, num_workers=4, shuffle=True, collate_fn=T.letterbox_collate_fn)

        model = GhostNano_3x3_m96(len(class_names)).eval()

        for images, target in dataloader:
            prediction = model(images)
            prediction = non_max_suppression(prediction, 0.25, 0.45)

            for i, image in enumerate(images):
                box_p = prediction[i][..., :4]
                lbl_p = prediction[i][..., 4:]
                lbl_p = [f'{class_names[int(x[1])]} {x[0]:.2f}' for x in lbl_p]
                box_t = target[target[..., 0] == i][..., 2:]
                lbl_t = target[target[..., 0] == i][..., 1]
                lbl_t = [class_names[int(x)] for x in lbl_t]
                canvas = V.Canvas(image)
                canvas.draw_boxes_with_label(box_p, lbl_p, color=(100, 149, 237))
                canvas.draw_boxes_with_label(box_t, lbl_t, color=(255, 165, 0))
                cv2.imwrite(f"test_image_{i}.png", canvas.image)
                logger.success(f"inference image at batch {i}")
            break

    except Exception as e:
        logger.error(e)
        raise e
