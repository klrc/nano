import cv2
import random
from matplotlib import pyplot as plt
import torch
import seaborn as sns

from torch.utils.data.dataloader import DataLoader


from nano.datasets.coco_box2d import MSCOCO
from nano.datasets.coco_box2d_transforms import Affine, Albumentations, SizeLimit, ToTensor, Mosaic4
from nano.datasets.coco_box2d_visualize import draw_bounding_boxes, draw_center_points, from_tensor_image
from nano.models.assigners.simota import SimOTA
from nano.datasets.coco_box2d import collate_fn


# preset configurations
img_root = "/home/sh/Datasets/coco3/images/val"
label_root = "/home/sh/Datasets/coco3/labels/val"
names = ["person", "bike", "car"]


def test_objectness(model, device):
    base = MSCOCO(imgs_root=img_root, annotations_root=label_root, min_size=416)
    base = SizeLimit(base, 5000)
    base = Affine(base, horizontal_flip=0.5, perspective=0.3, max_perspective=0.2)
    base = Albumentations(base, "random_blind")
    base = Mosaic4(base, img_size=448)
    trainset = ToTensor(base)

    train_loader = DataLoader(trainset, batch_size=1, num_workers=8, pin_memory=False, collate_fn=collate_fn)
    model.head.debug = True
    model = model.eval().to(device)

    i = 0
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        # targets = targets.to(device)
        with torch.no_grad():
            pred, grid_mask, stride_mask = model(imgs)
            # draw preds
            canvas = torch.zeros(imgs.shape[-2:]).to(device)
            pred = pred.flatten(0, 1)
            grid = (grid_mask[0] + 0.5) * stride_mask[0].unsqueeze(-1)
            for p, (gx ,gy) in zip(pred[:, 4], grid.int()):
                canvas[gy, gx] = p
            canvas /= canvas.max()
        imgs = (imgs[0]*0.5 + canvas.unsqueeze(0)*3).clamp(0, 1).cpu()
        cv_img = from_tensor_image(imgs)
        cv2.imwrite(f'test_activation_obj_{i}.png', cv_img)
        i += 1
        if i >= 4:
            return

if __name__ == "__main__":
    from nano.models.model_zoo.yolox_ghost import Ghostyolox_3x3_s32

    model = Ghostyolox_3x3_s32(num_classes=3)
    model.load_state_dict(torch.load("runs/train/exp18/last.pt")["state_dict"])
    model.train().to("cuda")

    test_objectness(model, "cuda")
