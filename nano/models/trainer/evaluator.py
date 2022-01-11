# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset
"""

import numpy as np
import torch
from torch.utils.data import dataloader
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from torchvision.ops import box_iou
from nano.models.trainer.trainer import load_device
from nano.ops.box2d import non_max_suppression
from nano.datasets.coco_box2d_metrics import ap_per_class


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


@torch.no_grad()
def run(model, class_names, dataloader, device, half=False, conf_thres=0.01, iou_thres=0.6):

    assert hasattr(model, "strides")

    # Half
    device = torch.device(device)
    half &= device.type != "cpu"  # half precision only supported on CUDA
    model.half() if half else model.float()

    # Configure
    model = model.eval().to(device)  # double check
    nc = len(class_names)  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Metrics
    seen = 0
    names = {k: v for k, v in enumerate(class_names)}
    s = ("%20s" + "%11s" * 6) % ("Class", "Images", "Labels", "P", "R", "mAP@.5", "mAP@.5:.95")
    p, r, f1, mp, mr, map50, map = [torch.zeros(1) for _ in range(7)]
    stats, ap, ap_class = [], [], []

    for img, targets in tqdm(dataloader, desc=s):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        targets = targets.to(device)

        # Run model
        out = model(img)  # inference and training outputs

        # Run NMS
        out = non_max_suppression(out, conf_thres, iou_thres)

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()  # native-space pred

            # Evaluate
            if nl:
                correct = process_batch(predn, labels, iouv)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = "%20s" + "%11i" * 2 + "%11.3g" * 4  # print format
    print(pf % ("all", seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Return results
    model.float()  # for training
    return mp, mr, map50, map


if __name__ == "__main__":
    from nano.datasets.coco_box2d import MSCOCO, collate_fn, letterbox_collate_fn
    from nano.models.model_zoo.nano_ghost import GhostNano_3x3_s32
    from nano.datasets.coco_box2d_transforms import (
        ToTensor,
    )
    from nano.models.trainer import load_device

    imgs_root = "/home/sh/Datasets/coco3/images/val"
    annotations_root = "/home/sh/Datasets/coco3/labels/val"
    base = MSCOCO(imgs_root=imgs_root, annotations_root=annotations_root, max_size=416)
    valset = ToTensor(base)

    batch_size = 64
    val_loader = DataLoader(valset, batch_size=batch_size // 2, num_workers=8, pin_memory=False, collate_fn=letterbox_collate_fn)

    model = GhostNano_3x3_s32(num_classes=3)
    model.eval()
    device = load_device("cuda")
    class_names = ["person", "bike", "car"]

    run(model, class_names, val_loader, device)
