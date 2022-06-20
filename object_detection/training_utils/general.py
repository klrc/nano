import glob
import inspect
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from loguru import logger
from torch.cuda import amp
from tqdm import tqdm

from .default_settings import DefaultSettings
from .ema import de_parallel
from .metrics import ap_per_class
from .yolov5_dataset_loader_pack import InfiniteDataLoader, LoadImagesAndLabels
from .yolov5_loss_func_pack.loss import yolov5_fake_hyp


FAST_DEBUG_MODE = False
if FAST_DEBUG_MODE:
    logger.warning("Warning: FAST DEBUG MODE on!")


class Status:
    # global status
    current_epoch = -1
    last_opt_step = -1

    # training status
    tr_accumulate = -1
    tr_nb = -1
    tr_nw = -1
    tr_loss_box = 0
    tr_loss_obj = 0
    tr_loss_cls = 0

    # validation status
    best_fitness = 0.0
    current_fitness = -1
    P = 0
    R = 0
    mAP_50 = 0
    mAP_95 = 0
    f1 = 0
    val_loss_box = 0
    val_loss_obj = 0
    val_loss_cls = 0


def sync_settings(config: DefaultSettings):
    if config is None:
        return DefaultSettings()
    assert isinstance(config, DefaultSettings)
    return config


def sync_status(train_loader, settings: DefaultSettings):
    s, u = settings, Status()
    u.tr_accumulate = max(round(s.nbs / s.batch_size), 1)  # accumulate loss before optimizing
    u.tr_nb = len(train_loader)  # number of batches
    u.tr_nw = max(round(s.warmup_epochs * u.tr_nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    return u


def sync_yolov5_hyp(model, settings: DefaultSettings):
    s = settings
    nl = de_parallel(model).detect.nl  # number of detection layers (to scale hyps)
    box = s.box * 3 / nl  # scale to layers
    cls = s.cls * s.nc / 80 * 3 / nl  # scale to classes and layers
    obj = s.obj * (s.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp = yolov5_fake_hyp(s.cls_pw, s.obj_pw, s.label_smoothing, s.fl_gamma, box, obj, cls, s.anchor_t)
    return hyp


def select_device(device="", batch_size=0, newline=True):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f"YOLOv5 🚀 torch {torch.__version__} "  # string
    device = str(device).strip().lower().replace("cuda:", "")  # to string, 'cuda:0' to '0'
    cpu = device == "cpu"
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(
            device.replace(",", "")
        ), f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(",") if device else "0"  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f"batch-size {batch_size} not multiple of GPU count {n}"
        space = " " * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
    else:
        s += "CPU\n"

    if not newline:
        s = s.rstrip()
    return torch.device("cuda:0" if cuda else "cpu")


def check_before_training(model: nn.Module, device, input_shape, expected_output_shapes=None):
    assert isinstance(model, nn.Module)
    model.to(device)
    try:
        test_tensor = torch.rand(*input_shape).to(device)
        model.train()
        model.forward(test_tensor)  # dry run
        model.eval()
        output = model.forward(test_tensor)
        eos = expected_output_shapes
        if eos is not None:
            if isinstance(output, torch.Tensor):
                output = (output,)
                eos = (eos,)
            assert len(eos) == len(output), f"eos length does not match with output.({len(eos)} vs. {len(output)})"
            for t1, t2 in zip(output, eos):
                assert t1.shape == t2.shape
        logger.success("model check passed")
        return model
    except Exception as e:
        logger.error("model inference check failed")
        raise e


def create_optimizer(model: nn.Module, frozen_params, optimizer_prototype, lr0, momentum, weight_decay):
    for k, v in model.named_parameters():
        v.requires_grad = True
        if frozen_params is not None and any(x == k for x in frozen_params):
            v.requires_grad = False

    pgroup = [[], [], []]  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):  # bias
            pgroup[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            pgroup[1].append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            pgroup[0].append(v.weight)

    for i in range(3):
        pgroup[i] = [x for x in pgroup[i] if x.requires_grad]

    op = optimizer_prototype
    if op is torch.optim.Adam or op is torch.optim.AdamW:
        optimizer = op(pgroup[2], lr=lr0, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif op is torch.optim.SGD:
        optimizer = op(pgroup[2], lr=lr0, momentum=momentum, nesterov=True)
    else:
        try:
            optimizer = op(pgroup[2], lr=lr0, momentum=momentum)
        except Exception as e:
            logger.error("unsupported optimizer args format.")
            raise e
    optimizer.add_param_group({"params": pgroup[0], "weight_decay": weight_decay})  # add g0 with weight_decay
    optimizer.add_param_group({"params": pgroup[1]})  # add g1 (BatchNorm2d weights)
    del pgroup
    return optimizer


def one_cycle_lambda(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def linear_lambda(lrf, epochs):
    return lambda x: (1 - x / epochs) * (1.0 - lrf) + lrf  # linear


def create_scheduler(optimizer, lrf, start_epoch, epochs, cos_lr=False):
    if cos_lr:
        lf = one_cycle_lambda(1, lrf, epochs)  # cosine 1->hyp['lrf']
    else:
        lf = linear_lambda(lrf, epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    scheduler.last_epoch = start_epoch - 1
    return scheduler, lf


def create_dataloader(dataset_path, training, settings: DefaultSettings):
    s = settings
    batch_size = s.batch_size
    workers = s.workers
    hyp = {
        "mosaic": s.mosaic,
        "mixup": s.mixup,
        "degrees": s.degrees,
        "translate": s.translate,
        "scale": s.scale,
        "shear": s.shear,
        "perspective": s.perspective,
        "hsv_h": s.hsv_h,
        "hsv_s": s.hsv_s,
        "hsv_v": s.hsv_v,
        "flipud": s.flipud,
        "fliplr": s.fliplr,
        "copy_paste": s.copy_paste,
    }
    if training:
        augment, shuffle, rect, pad = True, True, False, 0.0  # rect is incompatible with DataLoader shuffle
    else:
        augment, shuffle, rect, pad = False, False, True, 0.0
        batch_size *= 2
        workers *= 2
    dataset = LoadImagesAndLabels(
        dataset_path,
        s.imgsz,
        batch_size,
        augment=augment,
        hyp=hyp,
        rect=rect,
        cache_images=s.cache,
        stride=int(s.grid_stride),
        pad=pad,
        fake_osd=s.fake_osd,
        fake_darkness=s.fake_darkness,
    )
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    loader = InfiniteDataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=nw, sampler=None, pin_memory=True, collate_fn=LoadImagesAndLabels.collate_fn
    )
    return loader, dataset


def increment_path(path, exist_ok=False, sep=".", mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if (path.exists() or Path(f"{path}{sep}{0}").exists()) and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    else:
        path = Path(f"{path}{sep}{0}")
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y


def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, labels=(), max_det=300):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.1 + 0.03 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            logger.warning(f"WARNING: NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded
    return output


def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return float((x[:, :4] * w).sum(1))


def train_for_one_epoch(model: nn.Module, device, train_loader, optimizer, criteria, scaler, ema, scheduler, lf, settings: DefaultSettings, status: Status):
    s = settings
    u = status
    # training pipeline
    model.train()
    mloss = torch.zeros(3, device=device)  # mean losses
    pbar = enumerate(train_loader)
    print(("\n" + "%10s" * 5) % ("epoch", "gpu_mem", "box", "obj", "cls"))
    pbar = tqdm(pbar, total=u.tr_nb, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")  # progress bar
    optimizer.zero_grad()
    cuda = device.type != "cpu"

    for i, (imgs, targets, _, _) in pbar:  # batch -------------------------------------------------------------
        ni = i + u.tr_nb * u.current_epoch  # number integrated batches (since train start)
        imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

        # Warmup
        if ni <= u.tr_nw:
            xi = [0, u.tr_nw]  # x interp
            # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
            u.tr_accumulate = max(1, np.interp(ni, xi, [1, s.nbs / s.batch_size]).round())
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x["lr"] = np.interp(ni, xi, [s.warmup_bias_lr if j == 2 else 0.0, x["initial_lr"] * lf(u.current_epoch)])
                if "momentum" in x:
                    x["momentum"] = np.interp(ni, xi, [s.warmup_momentum, s.momentum])

        # Forward
        with amp.autocast(enabled=cuda):
            pred = model(imgs)  # forward
            loss, loss_items = criteria(pred, targets.to(device))  # loss scaled by batch_size

        # Backward
        scaler.scale(loss).backward()

        # Optimize
        if ni - u.last_opt_step >= u.tr_accumulate:
            scaler.step(optimizer)  # optimizer.step
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(model)
            u.last_opt_step = ni

        # Log
        mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
        mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
        pbar.set_description(("%10s" * 2 + "%10.4g" * 3) % (f"{u.current_epoch}/{s.max_epoch - 1}", mem, *mloss))
        u.tr_loss_box = float(mloss[0])
        u.tr_loss_obj = float(mloss[1])
        u.tr_loss_cls = float(mloss[2])

        if FAST_DEBUG_MODE:
            break

    scheduler.step()


def val_for_one_batch(detections, labels, iouv):
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
        matches = torch.from_numpy(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


@torch.no_grad()
def val_for_one_epoch(model, device, val_loader, criteria, settings: DefaultSettings, status: Status):
    s, u = settings, status
    cuda = device.type != "cpu"
    names = {k: v for k, v in enumerate(s.names)}
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    title = ("%20s" + "%11s" * 7) % ("class", "images", "labels", "P", "R", "mAP@.5", "mAP@.5:.95", "f1")
    p, r, mp, mr, map50, map, mf1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    stats = []
    ap_class = []
    seen = 0
    # validation pipeline
    model.eval()
    loss = torch.zeros(3, device=device)
    pbar = tqdm(val_loader, desc=title, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        if cuda:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if s.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        out, train_out = model(im)  # inference, loss outputs

        # Loss
        if s.compute_loss:
            loss += criteria([x.float() for x in train_out], targets)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        out = non_max_suppression(out, s.conf_thres, s.iou_thres, labels=[], multi_label=True, agnostic=False)

        # Metrics
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            shape = shapes[si][0]
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = val_for_one_batch(predn, labelsn, iouv)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

        if FAST_DEBUG_MODE:
            break

    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map, mf1 = p.mean(), r.mean(), ap50.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=s.nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = "%20s" + "%11i" * 2 + "%11.3g" * 5  # print format
    print(pf % ("all", seen, nt.sum(), mp, mr, map50, map, mf1))

    # Print results per class
    if s.verbose and s.nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i], f1[i]))

    # Print speeds
    # t = tuple(x / seen * 1e3 for x in dt)  # speeds per image

    # Return results
    model.float()  # for training
    maps = np.zeros(s.nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    # Update best mAP
    results = (mp, mr, map50, map, *(loss.cpu() / len(val_loader)).tolist())
    fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
    if fi > u.best_fitness:
        u.best_fitness = fi
    u.current_fitness = fi
    u.P = float(mp)
    u.R = float(mr)
    u.mAP_50 = float(map50)
    u.mAP_95 = float(map)
    u.f1 = float(mf1)
    if s.compute_loss:
        u.val_loss_box = float(results[4])
        u.val_loss_obj = float(results[5])
        u.val_loss_cls = float(results[6])


def save_data(data, path, mode="w"):
    if isinstance(data, DefaultSettings) or isinstance(data, Status):
        assert mode in ("w", "a")
        with open(path, mode) as f:
            for k, v in inspect.getmembers(data):
                if not k.startswith("__"):
                    line = f"{k}: {v}\n"
                    f.write(line)
            f.write("\n")
    else:
        torch.save(data, path)


def initialize_detect_bias(model, dataset, nc):
    cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.0
    model.detect._initialize_biases(cf)


def convert_dataset_labels(label_root, mapper, dataset_dirs=("train2017", "val2017"), backup_suffix="_raw"):
    # raw backup
    backup_root = label_root + backup_suffix
    if os.path.exists(backup_root):
        os.system(f"rm -r {label_root}")
    else:
        os.system(f"mv {label_root} {backup_root}")
    # map label classses
    for dataset in dataset_dirs:
        source_dir = f"{backup_root}/{dataset}"
        output_dir = f"{label_root}/{dataset}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # traverse through images
        print(source_dir)
        assert os.path.exists(source_dir)
        for file_name in os.listdir(source_dir):
            if not file_name.startswith("._") and file_name.endswith(".txt"):
                labels = []
                with open(f"{source_dir}/{file_name}", "r") as f:
                    for line in f.readlines():
                        c, x, y, w, h = [float(m) for m in line.split(" ")]
                        c = int(c)
                        if c in mapper:
                            c = mapper[c]
                            line = f"{c} {x} {y} {w} {h}"
                            labels.append(line)

                with open(f"{output_dir}/{file_name}", "w") as f:
                    for line in labels:
                        f.write(f"{line}\n")
