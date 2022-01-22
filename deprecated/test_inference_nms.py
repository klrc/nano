import cv2
import torch

from torch.utils.data.dataloader import DataLoader

from nano.datasets.coco_box2d import MSCOCO
from nano.datasets.coco_box2d_transforms import SizeLimit, ToTensor
from nano.datasets.coco_box2d_visualize import draw_bounding_boxes
from nano.datasets.coco_box2d import letterbox_collate_fn
from nano.ops.box2d import non_max_suppression


# preset configurations
img_root = "../datasets/coco3/images/val"
label_root = "../datasets/coco3/labels/val"
names = ["person", "bike", "car"]


def test_nms(model, device, focal_nms=True):
    base = MSCOCO(imgs_root=img_root, annotations_root=label_root, max_size=448)
    base = SizeLimit(base, 100)
    trainset = ToTensor(base)

    train_loader = DataLoader(trainset, batch_size=1, num_workers=8, pin_memory=False, collate_fn=letterbox_collate_fn)
    model.head.debug = False
    model = model.eval().to(device)

    i = 0
    for img, _ in train_loader:
        img = img.to(device)
        # targets = targets.to(device)
        with torch.no_grad():

            pred = model(img)
            pred = non_max_suppression(pred, 0.25, 0.45, focal_nms=focal_nms)[0]

            # draw lobj centers
            box_pred = pred[:, :4]
            alp_pred = pred[:, 4]
            cls_pred = pred[:, 5]
            label_pred = [f'{names[int(x)]} conf={alp_pred[ia]:.2f}' for ia, x in enumerate(cls_pred.cpu())]

            cv_img = img[0].cpu() * 0.4
            cv_img = draw_bounding_boxes(cv_img, boxes=box_pred.cpu(), boxes_label=label_pred, alphas=alp_pred)
            cv2.imwrite(f"test_{'focal_'if focal_nms else ''}nms_{i}.png", cv_img)

        i += 1
        if i >= 20:
            return


if __name__ == "__main__":
    from nano.models.model_zoo.nano_ghost import GhostNano_3x3_m96

    model = GhostNano_3x3_m96(num_classes=3)
    model.load_state_dict(torch.load("release/GhostNano_3x3_m96/GhostNano_3x3_m96.pt", map_location="cpu"))
    model.train().to("cpu")

    test_nms(model, 'cpu', focal_nms=True)
    test_nms(model, 'cpu', focal_nms=False)
