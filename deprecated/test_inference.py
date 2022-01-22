import cv2
import torch

from torch.utils.data.dataloader import DataLoader

from nano.datasets.coco_box2d import MSCOCO
from nano.datasets.coco_box2d_transforms import SizeLimit, ToTensor
from nano.datasets.coco_box2d_visualize import draw_bounding_boxes
from nano.datasets.coco_box2d import letterbox_collate_fn


# preset configurations
img_root = "../datasets/coco3/images/val"
label_root = "../datasets/coco3/labels/val"
names = ["person", "bike", "car"]


def test_prediction(model, device):
    base = MSCOCO(imgs_root=img_root, annotations_root=label_root, max_size=448)
    base = SizeLimit(base, 100)
    trainset = ToTensor(base)

    train_loader = DataLoader(trainset, batch_size=1, num_workers=8, pin_memory=False, collate_fn=letterbox_collate_fn)
    model.head.debug = True
    model = model.eval().to(device)

    i = 0
    for img, _ in train_loader:
        img = img.to(device)
        # targets = targets.to(device)
        with torch.no_grad():
            pred, grid_mask, stride_mask = model(img)
            pred, grid_mask, stride_mask = pred[0], grid_mask[0], stride_mask[0]

            # draw lobj centers
            box_pred = pred[:, :4]
            cls_pred = pred[:, 4:]
            centers = (grid_mask + 0.5) * stride_mask.unsqueeze(-1)
            alphas = cls_pred.max(dim=-1).values
            mask = alphas > 0.25
            box_pred, cls_pred, centers, alphas = box_pred[mask], cls_pred[mask], centers[mask], alphas[mask]
            if len(alphas) > 0:
                alphas /= alphas.max()
            label_pred = torch.argmax(cls_pred, dim=1, keepdim=True)
            label_pred = [names[x] for x in label_pred.cpu()]

            cv_img = img[0].cpu() * 0.4
            cv_img = draw_bounding_boxes(cv_img, boxes=box_pred.cpu(), boxes_label=label_pred, boxes_centers=centers, alphas=alphas)
            # cv_img = draw_center_points(cv_img, centers, thickness=3, alphas=alphas)
            cv2.imwrite(f"test_prediction_{i}.png", cv_img)

        i += 1
        if i >= 20:
            return


if __name__ == "__main__":
    from nano.models.model_zoo.nano_ghost import GhostNano_3x4_m96

    model = GhostNano_3x4_m96(num_classes=3)
    model.load_state_dict(torch.load("release/GhostNano_3x4_m96/GhostNano_3x4_m96.pt", map_location="cpu"))
    model.train().to("cpu")

    test_prediction(model, "cpu")
