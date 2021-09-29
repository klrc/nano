
import pytorch_lightning as pl
import torch.nn as nn
import torch
import math


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


class Loss(nn.Module):
    def __init__(self, hyp, nc=80, imgsz=416, anchors=([10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326])):
        super().__init__()
        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = self.smooth_BCE(eps=0)  # positive, negative BCE targets

        # Define criteria
        self.balance = {3: [4.0, 1.0, 0.4]}.get(3, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.BCEcls = nn.BCEWithLogitsLoss()
        self.BCEobj = nn.BCEWithLogitsLoss()

        # Detection args
        self.gr = 1.0
        self.nc = nc                     # num_classes
        self.nl = len(anchors)           # num_detection_layers
        self.na = len(anchors[0]) // 2   # num_anchors
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.hyp = hyp

        # Model parameters
        hyp['box'] *= 3. / self.nl  # scale to layers
        hyp['cls'] *= nc / 80. * 3. / self.nl  # scale to classes and layers
        hyp['obj'] *= (imgsz / 640) ** 2 * 3. / self.nl  # scale to image size and layers

    @staticmethod
    def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
        # return positive, negative label smoothing BCE targets
        return 1.0 - 0.5 * eps, 0.5 * eps

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE-target

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


class Shell(pl.LightningModule):
    def __init__(self, model, loss_fn, val_fn, device, hyp):
        super().__init__()
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.val_fn = val_fn.to(device)
        self.best_fitness = 0
        self.hyp = hyp

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        result = self.model(x)
        return result

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        imgs, targets, _, _ = batch
        imgs = imgs.float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
        pred = self.model(imgs)  # forward
        loss, loss_items = self.loss_fn(pred, targets)  # loss scaled by batch_size
        # Logging to TensorBoard by default
        self.log("train_loss/lbox", loss_items[0])
        self.log("train_loss/lobj", loss_items[1])
        self.log("train_loss/lcls", loss_items[2])
        return loss

    def on_epoch_end(self) -> None:
        # evaluation at the end of an epoch.
        with torch.no_grad():
            fitness, metrics = self.val_fn(self.model)  # loss scaled by batch_size
        self.model.train()
        # Logging to TensorBoard by default
        for k, v in metrics.items():
            self.log(k, v)
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            # save model
            torch.save(self.model.state_dict(), f'yolov5_shufflenet_1_5x_e{self.current_epoch()}.pt')

    def configure_optimizers(self):
        hyp = self.hyp
        # Optimizer
        g0, g1, g2 = [], [], []
        for v in self.model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                g2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d):  # weight with decay
                g0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight without decay
                g1.append(v.weight)

        optimizer = torch.optim.SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
        optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})
        optimizer.add_param_group({'params': g2})
        del g0, g1, g2
        # LR Scheduler
        lf = one_cycle(1, hyp['lrf'], 300)  # max epochs
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        return {'optimizer': optimizer, 'scheduler': scheduler}
