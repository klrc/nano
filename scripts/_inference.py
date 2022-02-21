import torch
import cv2
from loguru import logger
from nano.data.dataset_info import coco_to_drive3, voc_to_drive3
from nano.data.visualize import Canvas
import nano.data.transforms as T
from nano.models.box2d import non_max_suppression


def test_inference(model, device):
    names = ["person", "bike", "car"]
    valset = detection_data_layer("../datasets/VOC/images/val2012", "../datasets/VOC/labels/val2012")
    factory = Assembly()
    factory.append_data(valset)
    factory.compose(
        T.ToNumpy(mark_source=True),
        T.IndexMapping(coco_to_drive3, pattern="/coco"),
        T.IndexMapping(voc_to_drive3, pattern="/VOC"),
        T.Resize(max_size=416),
        T.ToTensor(),
    )
    valloader = factory.as_dataloader(batch_size=1, num_workers=4, collate_fn=T.letterbox_collate_fn)
    model.head.debug = False
    model = model.eval().to(device)

    for i, (img, _) in enumerate(valloader):
        img = img.to(device)
        with torch.no_grad():
            pred = model(img)
            pred = non_max_suppression(pred, 0.25, 0.45)[0]
            # draw lobj centers
            box_pred = pred[:, :4]
            alp_pred = pred[:, 4]
            cls_pred = pred[:, 5]
            label_pred = [f"{names[int(x)]} conf={alp_pred[ia]:.2f}" for ia, x in enumerate(cls_pred.cpu())]
            cv_img = img[0].cpu() * 0.4
            canvas = Canvas(cv_img)
            for box, label, alpha in zip(box_pred, label_pred, alp_pred):
                box = [int(x) for x in box]
                alpha = float(alpha)
                canvas.draw_box(box, alpha)
                canvas.draw_text_with_background(label, (box[0], box[1]), alpha)
                canvas.next_color()
            cv2.imwrite(f"test_inference_{i}.png", canvas.image)
            logger.success(f"saved output as test_inference_{i}.png")
        if i > 4:
            break
