# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset
"""

import math
import os
import random
from copy import deepcopy
from pathlib import Path
import wandb
import numpy as np
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from tqdm import tqdm
import evaluator
import glob
import re



def load_device(device="cuda"):
    cpu = device == "cpu"
    if cpu:
        # force torch.cuda.is_available() = False
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:  # non-cpu device requested
        # check availability
        assert torch.cuda.is_available(), f"CUDA unavailable, invalid device {device} requested"
    return torch.device(device)


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix("")
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == "" else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum()


class ModelEMA:
    """Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        if next(model.parameters()).device.type != "cpu":
            self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)


class EarlyStopping:
    # YOLOv5 simple early stopper
    def __init__(self, patience=30):
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float("inf")  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        return stop


def run(
    model,
    train_loader,
    val_loader,
    class_names,
    criteria,
    device,
    batch_size=32,
    patience=10,
    epochs=300,
    lr0=0.0032,
    momentum=0.843,
    weight_decay=0.00036,
    lrf=0.12,
    optimizer='SGD',
    ckpt=None,
    warmup_epochs=3,
    warmup_bias_lr=0.05,
    warmup_momentum=0.5,
    load_optimizer=False,
    verbose=True,
):
    if verbose:
        logger = wandb.init(project="nano", dir="./runs")

    # Directories
    w = increment_path(Path("runs/train") / "exp", exist_ok=False)
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / "last.pt", w / "best.pt"

    # Config
    cuda = device.type != "cpu"
    init_seeds(0)

    # Model
    assert isinstance(model, torch.nn.Module)
    model.to(device)

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    weight_decay *= batch_size * accumulate / nbs  # scale weight_decay

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    if optimizer == 'Adam':
        optimizer = Adam(g0, lr=lr0, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif optimizer == 'AdamW':
        optimizer = AdamW(g0, lr=lr0, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif optimizer == 'SGD':
        optimizer = SGD(g0, lr=lr0, momentum=momentum, nesterov=True)

    optimizer.add_param_group({"params": g1, "weight_decay": weight_decay})  # add g1 with weight_decay
    optimizer.add_param_group({"params": g2})  # add g2 (biases)
    del g0, g1, g2

    # Scheduler
    lf = one_cycle(1, lrf, epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model)

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if ckpt is not None:
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt.pop("state_dict"))
        if "optimizer" in ckpt:
            optimizer_state_dict = ckpt.pop("optimizer")
            if load_optimizer:
                optimizer.load_state_dict(optimizer_state_dict)
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"]
        if verbose:
            logger.log(ckpt)

    # Anchors
    model.half().float()  # pre-reduce anchor precision

    # Start training
    nb = len(train_loader)  # number of batches
    nw = max(round(warmup_epochs * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    stopper = EarlyStopping(patience=patience)

    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):  # epoch ------------------------------------------------------------------
        model.train()

        mloss = torch.zeros(3, device=device)  # mean losses
        pbar = enumerate(train_loader)
        print(("\n" + "%10s" * 7) % ("Epoch", "gpu_mem", "box", "obj", "cls", "labels", "topk"))
        pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True)

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [warmup_bias_lr if j == 2 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [warmup_momentum, momentum])

            # Forward
            with amp.autocast(enabled=cuda):
                result = model(imgs)  # forward
                loss, loss_items = criteria(result, targets.to(device))  # loss scaled by batch_size

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
            pbar.set_description(("%10s" * 2 + "%10.4g" * 5) % (f"{epoch}/{start_epoch + epochs - 1}", mem, *mloss, targets.shape[0], criteria._avg_topk))

            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        # lr = [x["lr"] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        # mAP
        ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])

        # Calculate mAP
        results = evaluator.run(
            model=ema.ema,
            class_names=class_names,
            dataloader=val_loader,
            device=device,
            half=False,
        )

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        if fi > best_fitness:
            best_fitness = fi

        # Upload logger
        log_vals = {
            "epoch": epoch,
            "precision": results[0].item(),
            "recall": results[1].item(),
            "map@.5": results[2].item(),
            "mAP@.5-.95": results[3].item(),
            "train_loss_box": mloss[0].item(),
            "train_loss_obj": mloss[1].item(),
            "train_loss_cls": mloss[2].item(),
            "train_loss": mloss.sum().item(),
        }
        if logger is not None:
            logger.log(log_vals)

        # Save model
        ckpt = {
            "state_dict": deepcopy(ema.ema).half().state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        ckpt.update(log_vals)

        # Save last, best and delete
        torch.save(ckpt, last)
        if best_fitness == fi:
            torch.save(ckpt, best)
        del ckpt

        # Stop Single-GPU
        if stopper(epoch=epoch, fitness=fi):
            break

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    torch.cuda.empty_cache()
    return best


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from nano.models.assigners.simota import (
        SimOTA,
        AssignGuidanceSimOTA,
    )
    from nano.models.model_zoo.yolox_ghost import (
        Ghostyolox_3x3_s32,
        Ghostyolox_3x3_m48,
        Ghostyolox_4x3_l128,
    )
    from nano.models.model_zoo.yolox_es import ESyolox_3x3_m96
    from nano.datasets.coco_box2d import MSCOCO, collate_fn, letterbox_collate_fn
    from nano.datasets.coco_box2d_transforms import (
        SizeLimit,
        Affine,
        Albumentations,
        Mosaic4,
        ToTensor,
    )

    imgs_root = "/home/sh/Datasets/coco3/images/train"
    annotations_root = "/home/sh/Datasets/coco3/labels/train"
    base = MSCOCO(imgs_root=imgs_root, annotations_root=annotations_root, min_size=416)
    base = SizeLimit(base, 30000)
    base = Affine(base, horizontal_flip=0.3, perspective=0.3, max_perspective=0.2)
    base = Albumentations(base, "random_blind")
    base = Mosaic4(base, img_size=448)
    trainset = ToTensor(base)
    imgs_root = "/home/sh/Datasets/coco3/images/val"
    annotations_root = "/home/sh/Datasets/coco3/labels/val"
    base = MSCOCO(imgs_root=imgs_root, annotations_root=annotations_root, max_size=416)
    valset = ToTensor(base)

    batch_size = 32
    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=8, pin_memory=False, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(valset, batch_size=batch_size//2, num_workers=8, pin_memory=False, collate_fn=letterbox_collate_fn)

    model = Ghostyolox_3x3_s32(num_classes=3)
    device = load_device("cuda")
    class_names = ["person", "bike", "car"]

    criteria = SimOTA(3, True)

    run(
        model,
        train_loader,
        val_loader,
        class_names,
        criteria,
        device,
        lr0=0.001,
        optimizer='SGD',
        warmup_epochs=3,
        batch_size=batch_size,
        patience=16,
        epochs=300,
    )
