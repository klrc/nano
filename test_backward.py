import cv2
import random
import torch

from torch.utils.data.dataloader import DataLoader


from nano.datasets.coco_box2d import MSCOCO
from nano.datasets.coco_box2d_transforms import Affine, Albumentations, SizeLimit, ToTensor, Mosaic4
from nano.datasets.coco_box2d_visualize import draw_bounding_boxes
from nano.models.assigners.simota import SimOTA
from nano.datasets.coco_box2d import collate_fn


# preset configurations
img_root = "/home/sh/Datasets/coco3/images/val"
label_root = "/home/sh/Datasets/coco3/labels/val"
names = ["person", "bike", "car"]
    

def test_backward(model, device):
    base = MSCOCO(imgs_root=img_root, annotations_root=label_root, min_size=416)
    base = SizeLimit(base, 20000)
    base = Affine(base, horizontal_flip=0.5, perspective=0.3, max_perspective=0.2)
    base = Albumentations(base, "random_blind")
    base = Mosaic4(base, img_size=448)
    trainset = ToTensor(base)

    train_loader = DataLoader(trainset, batch_size=32, num_workers=8, pin_memory=False, collate_fn=collate_fn)
    assigner = SimOTA(num_classes=3, with_loss=True)

    for imgs, targets in train_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        with torch.autograd.set_detect_anomaly(True):
            result = model(imgs)
            loss, detached_loss = assigner(result, targets)
            print(detached_loss)
            loss.backward()
        return


def test_assignment(model, device):
    imgs_root = "/home/sh/Datasets/coco3/images/train"
    annotations_root = "/home/sh/Datasets/coco3/labels/train"
    base = MSCOCO(imgs_root=imgs_root, annotations_root=annotations_root, min_size=416)
    base = SizeLimit(base, 20000)
    base = Affine(base, horizontal_flip=0.5, perspective=0.3, max_perspective=0.2)
    base = Albumentations(base, "random_blind")
    base = Mosaic4(base, img_size=448)
    trainset = ToTensor(base)

    train_loader = DataLoader(trainset, batch_size=1, num_workers=8, pin_memory=False, collate_fn=collate_fn)
    assigner = SimOTA(num_classes=3, with_loss=False)

    i = 0
    for imgs, targets in train_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        with torch.autograd.set_detect_anomaly(True):
            result = model(imgs)
            pred, grid_mask, stride_mask = result
            reg_targets, obj_targets, cls_targets = assigner(result, targets)
            obj_targets = obj_targets.bool()
            cls_targets = torch.argmax(cls_targets, dim=1, keepdim=True)
            center_targets = (grid_mask[0, obj_targets] + 0.5) * stride_mask[0, obj_targets].unsqueeze(-1)
            for b, c in zip(reg_targets, center_targets):
                print(b, c)
            # draw targets
            cv_img = draw_bounding_boxes(imgs[0].cpu(), torch.cat((cls_targets, reg_targets), -1).cpu(), names, centers=center_targets.cpu())
            cv2.imwrite(f"test_assignment_{i}.png", cv_img)
            # draw preds
            pred = pred.flatten(0, 1)[obj_targets]
            pred_cls = torch.argmax(pred[:, 5:], dim=1, keepdim=True)
            pred_box = pred[:, :4]
            matched_pred = torch.cat((pred_cls, pred_box), dim=-1)
            src_copy = cv_img.copy()
            cv_img = draw_bounding_boxes(cv_img, matched_pred, names, box_color=(0,140,255))
            cv_img = cv2.addWeighted(src_copy, 0.6, cv_img, 0.4, 1)
            cv2.imwrite(f"test_assignment_dense_{i}.png", cv_img)
        i += 1
        if i >= 4:
            return


if __name__ == "__main__":
    from nano.models.model_zoo.yolox_ghost import Ghostyolox_3x3_s32

    model = Ghostyolox_3x3_s32(num_classes=3)
    model.train().to("cuda")
    # test_assignment(model, "cuda")
    test_backward(model, 'cuda')
