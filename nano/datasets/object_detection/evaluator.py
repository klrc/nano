from matplotlib.pyplot import cla
import torch
import torch.nn as nn
import numpy as np

from .metrics import ap_per_class, fitness
from .general import non_max_suppression, scale_coords, xywh2xyxy, box_iou


def process_batch(predictions, labels, iouv):
    # Evaluate 1 batch of predictions
    correct = torch.zeros(predictions.shape[0], len(iouv), dtype=torch.bool, device=iouv.device)
    detected = []  # label indices
    tcls, pcls = labels[:, 0], predictions[:, 5]
    nl = labels.shape[0]  # number of labels
    for cls in torch.unique(tcls):
        ti = (cls == tcls).nonzero().view(-1)  # label indices
        pi = (cls == pcls).nonzero().view(-1)  # prediction indices
        if pi.shape[0]:  # find detections
            ious, i = box_iou(predictions[pi, 0:4], labels[ti, 1:5]).max(1)  # best ious, indices
            detected_set = set()
            for j in (ious > iouv[0]).nonzero():
                d = ti[i[j]]  # detected label
                if d.item() not in detected_set:
                    detected_set.add(d.item())
                    detected.append(d)  # append detections
                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                    if len(detected) == nl:  # all labels already located in image
                        break
    return correct

class CallmAP(nn.Module):
    def __init__(self, names, conf_thres=0.25, iou_thres=0.45):
        super().__init__()
        self.names = {i:n for i, n in enumerate(names)}
        self.conf_thres=conf_thres  # confidence threshold
        self.iou_thres=iou_thres    # NMS IOU threshold

    def forward(self, imgs, targets, out, shapes):
        device = imgs.device()
        iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        stats, ap, ap_class = [], [], []
        _, _, height, width = imgs.shape  # batch size, channels, height, width
        names = self.names
        nc = len(names)

        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = []  # for autolabelling
        out = non_max_suppression(out, self.conf_thres, self.iou_thres, labels=lb, multi_label=True, agnostic=False)

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            shape = shapes[si][0]

            # Predictions
            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
            predn = pred.clone()
            scale_coords(imgs[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(imgs[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)
        fi = fitness(np.array([mp, mr, map50, map]).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        metrics = {
            'num_targets': nt,
            'mean_precision': mp,
            'mean_recall': mr,
            'mAP:.5': map50,
            'mAP': map,
        }
        if nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                metrics[f'num_targets@{names[c]}'] = nt[c]
                metrics[f'mean_precision@{names[c]}'] = p[i],
                metrics[f'mean_recall@{names[c]}'] = r[i],
                metrics[f'mAP:.5@{names[c]}'] = ap50[i],
                metrics[f'mAP@{names[c]}'] = ap[i],
        return fi, metrics