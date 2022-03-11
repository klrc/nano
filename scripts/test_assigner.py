import sys
from loguru import logger
import torch
import cv2

sys.path.append(".")


if __name__ == "__main__":
    try:
        logger.debug("Assigner Test ------------------")
        import nano.data.transforms as T
        import nano.data.visualize as V
        from nano.data.dataset import voc_quick_test_preset
        from nano.models.assigners.simota import SimOTA
        from nano.models.model_zoo.nano_ghost import GhostNano_3x3_m96

        class_names = "person|bike|motorcycle|car|bus|truck|OOD".split("|")
        dataset = voc_quick_test_preset((224, 416), class_names, "/Volumes/ASM236X")
        dataloader = dataset.as_dataloader(batch_size=16, num_workers=4, shuffle=True, collate_fn=T.letterbox_collate_fn)
        assigner = SimOTA(class_balance=[0.3, 1, 1, 1, 1, 1, 1])
        model = GhostNano_3x3_m96(len(class_names)).train()

        # test for one batch
        for images, target in dataloader:
            logger.info("Model forward")
            pred = model(images)

            logger.info("Backward test")
            with torch.autograd.set_detect_anomaly(True):
                loss, loss_items = assigner.forward(pred, target, debug=False)
                logger.success(loss_items)
                loss.backward()

            logger.info("Assignment test")
            boxes = assigner.forward(pred, target, debug=True)
            for i, (im, (bp, bt)) in enumerate(zip(images, boxes)):
                logger.success(f"Rendering batch {i}")
                canvas = V.Canvas(im)
                canvas.draw_boxes(bp, alpha=0.6, color=(100, 149, 237))
                canvas.draw_boxes(bt, thickness=1, color=(255, 165, 0))
                cv2.imwrite(f"test_image_{i}.png", canvas.image)
            break

    except Exception as e:
        logger.error(e)
        raise e
