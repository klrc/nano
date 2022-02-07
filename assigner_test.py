from nano.data.visualize import Canvas, RenderLabels
from nano.data.dataset import detection_data_layer, Assembly
from nano.data.dataset_info import voc_to_drive3, coco_to_drive3, drive3_names
from nano.models.model_zoo import GhostNano_3x3_s64
import nano.data.transforms as T
import nano.data.visualize as V
import cv2
import torch
from loguru import logger

from nano.models.assigners.simota import SimOTA
from nano.models.model_zoo.nano_ghost import GhostNano_3x4_m96

if __name__ == "__main__":

    try:
        logger.debug("Assigner Test ------------------")

        dataset1 = detection_data_layer("../datasets/coco/images/train2017", "../datasets/coco/labels/train2017")
        dataset2 = detection_data_layer("../datasets/VOC/images/train2012", "../datasets/VOC/labels/train2012")
        factory = Assembly()
        factory.append_data(dataset1, dataset2)
        factory.compose(
            T.ToNumpy(mark_source=True),
            T.IndexMapping(coco_to_drive3, pattern="/coco"),
            T.IndexMapping(voc_to_drive3, pattern="/VOC"),
            T.HorizontalFlip(p=0.5),
            T.Resize(max_size=736),
            T.RandomScale(min_scale=0.75, max_scale=1.25),
            T.RandomAffine(min_scale=0.75, max_scale=1.25, p=0.5),
            T.HSVTransform(),
            T.AlbumentationsPreset("random_blind"),
            T.Mosaic4(mosaic_size=736, p=0.5),
            T.ToTensor(),
        )
        trainloader = factory.as_dataloader(batch_size=1, num_workers=1, collate_fn=T.letterbox_collate_fn)
        assigner = SimOTA(num_classes=len(drive3_names), compute_loss=True)
        model = GhostNano_3x4_m96(len(drive3_names))
        model.train()

        logger.debug("backward test")

        for imgs, targets in trainloader:
            with torch.autograd.set_detect_anomaly(True):
                result = model(imgs)
                loss, detached_loss = assigner(result, targets)
                loss.backward()
                logger.success(f"loss = {detached_loss}")
                break

        logger.debug("assignment test")

        assigner.compute_loss = False
        i = 0
        for img, target in trainloader:
            with torch.autograd.set_detect_anomaly(True):
                result = model(img)
                p, gm, sm = result
                mm, box_t, obj_t, cls_t = assigner(result, target)
                nx5_target = torch.cat((cls_t[mm].unsqueeze(-1), box_t), dim=-1)
                c = (gm[0, mm] + 0.5) * sm[0, mm].unsqueeze(-1)

                # draw matched targets
                image = RenderLabels.functional(img[0], nx5_target, drive3_names)
                canvas = Canvas(image)
                for x, y in c:
                    canvas.draw_point((int(x), int(y)))
                    canvas.next_color()

                cv2.imwrite(f"test_assignment_box_{i}.png", canvas.image)
                logger.success(f"saved output as test_assignment_box_{i}.png")
            i += 1
            if i >= 4:
                break

    except Exception as e:
        logger.error(e)
