import os
import random
from loguru import logger
import torch
import numpy as np


import math
from copy import deepcopy
from pathlib import Path
import wandb
import torch.nn as nn
from torch.cuda import amp
from torch.optim import SGD, Adam, AdamW, lr_scheduler
import glob
import re

from torchvision.ops import box_iou
from nano.models.multiplex.box2d import non_max_suppression


def load_device(device="cuda"):
    cpu = device == "cpu"
    if cpu:
        # force torch.cuda.is_available() = False
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return torch.device("cpu")
    else:  # non-cpu device requested
        # check availability
        if ":" in device:
            device_id = device.split(":")[-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        assert torch.cuda.is_available(), f"CUDA unavailable, invalid device {device} requested"
    return torch.device("cuda")


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


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False):
    """Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    # if plot:
    #     plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
    #     plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
    #     plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
    #     plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype("int32")


def compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def val_batch(detections, labels, iouv):
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


class Trainer:
    def __init__(
        self,
        dataloader,
        model,
        criteria,
        device,
        batch_size=64,
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        lrf=0.1,
        optimizer="AdamW",
        warmup_epochs=3,
        warmup_bias_lr=0.1,
        warmup_momentum=0.5,
        start_epoch=0,
        end_epoch=300,
    ):
        self.check_device(model, device)
        self.init_optimizer(optimizer, lr0, weight_decay, momentum, batch_size, warmup_epochs, warmup_bias_lr, warmup_momentum)
        self.init_scheduler(lrf, start_epoch, end_epoch)
        self.init_ema()
        self.init_dataloader(dataloader)
        self.init_criteria(criteria)
        self.init_scaler()

    def check_device(self, model, device):
        # Config
        assert isinstance(model, torch.nn.Module)
        self.model = model
        self.device = load_device(device)
        self.cuda = self.device.type != "cpu"
        init_seeds(0)
        logger.success("Device checked")

    def init_optimizer(
        self,
        optimizer,
        lr0,
        weight_decay,
        momentum,
        batch_size,
        warmup_epochs,
        warmup_bias_lr,
        warmup_momentum,
    ):
        # Optimizer
        self.nbs = nbs = 64  # nominal batch size
        accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
        weight_decay *= batch_size * accumulate / nbs  # scale weight_decay

        self.accumulate = accumulate
        self.momentum = momentum
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.warmup_bias_lr = warmup_bias_lr
        self.warmup_momentum = warmup_momentum

        g0, g1, g2 = [], [], []  # optimizer parameter groups
        for v in self.model.modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):  # bias
                g2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
                g0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g1.append(v.weight)
        g0 = filter(lambda x: x.requires_grad, g0)
        g1 = filter(lambda x: x.requires_grad, g1)
        g2 = filter(lambda x: x.requires_grad, g2)

        if optimizer == "Adam":
            optimizer = Adam(g0, lr=lr0, betas=(momentum, 0.999))  # adjust beta1 to momentum
        elif optimizer == "AdamW":
            optimizer = AdamW(g0, lr=lr0, betas=(momentum, 0.999))  # adjust beta1 to momentum
        elif optimizer == "SGD":
            optimizer = SGD(g0, lr=lr0, momentum=momentum, nesterov=True)

        optimizer.add_param_group({"params": g1, "weight_decay": weight_decay})  # add g1 with weight_decay
        optimizer.add_param_group({"params": g2})  # add g2 (biases)
        del g0, g1, g2
        self.optimizer = optimizer
        logger.success(f"Init optimizer as {self.optimizer}")

    def init_scheduler(self, lrf, start_epoch, end_epoch):
        # Scheduler
        self.lf = one_cycle(1, lrf, end_epoch)  # cosine 1->hyp['lrf']
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
        self.scheduler.last_epoch = start_epoch - 1  # do not move
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        logger.success(f"Init scheduler as {self.scheduler}")

    def init_ema(self):
        # EMA
        self.model.half().float()  # pre-reduce anchor precision
        self.model.to(self.device)
        self.ema = ModelEMA(self.model)
        logger.success(f"Init model EMA as {self.ema}")

    def init_scaler(self):
        self.scaler = amp.GradScaler(enabled=self.cuda)
        logger.success(f"Init amp.GradScaler with self.cuda = {self.cuda}")

    def init_dataloader(self, dataloader):
        self.dataloader = dataloader
        self.iters = len(dataloader)
        self.nw = max(round(self.warmup_epochs * self.iters), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
        self.last_opt_step = -1
        logger.success(f"Init dataloader with {len(self.dataloader)} batches x {self.batch_size} batch size")

    def init_criteria(self, criteria):
        self.criteria = criteria
        logger.success(f"Init criteria as {self.criteria}")

    def train_for_one_epoch(self, epoch):
        # init model state
        self.model.train().half().float()
        mloss = torch.zeros(2, device=self.device)  # mean losses
        logger.info(("%10s" * 7) % ("Epoch", "gpu_mem", "box", "quality", "lr0", "lr1", "lr2"))
        self.optimizer.zero_grad()

        for i, (imgs, targets) in enumerate(self.dataloader):  # batch -------------------------------------------------------------
            ni = i + self.iters * epoch  # number integrated batches (since train start)
            imgs = imgs.to(self.device, non_blocking=True)

            # Warmup
            if ni <= self.nw:
                xi = [0, self.nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                self.accumulate = max(1, np.interp(ni, xi, [1, self.nbs / self.batch_size]).round())
                for j, x in enumerate(self.optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [self.warmup_bias_lr if j == 2 else 0.0, x["initial_lr"] * self.lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [self.warmup_momentum, self.momentum])

            # Forward
            with amp.autocast(enabled=self.cuda):
                result = self.model(imgs)  # forward
                loss, loss_items = self.criteria(result, targets.to(self.device))  # loss scaled by batch_size

            # Backward
            self.scaler.scale(loss).backward()

            # Optimize
            if ni - self.last_opt_step >= self.accumulate:
                self.scaler.step(self.optimizer)  # optimizer.step
                self.scaler.update()
                self.optimizer.zero_grad()
                self.ema.update(self.model)
                self.last_opt_step = ni

            # Log
            lr_g0, lr_g1, lr_g2 = [x["lr"] for x in self.optimizer.param_groups]  # for loggers
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
            if i % 10 == 0:
                logger.info(("%10s" * 2 + "%10.4f" * 5) % (f"{epoch}.{i}/{self.end_epoch}", mem, *mloss, lr_g0, lr_g1, lr_g2))
            # end batch ------------------------------------------------------------------------------------------------
        # Scheduler
        self.scheduler.step()
        self.ema.update_attr(self.model)
        return mloss


class Validator:
    def __init__(self, dataloader, class_names, device, half=True, conf_thres=0.01, iou_thres=0.6):
        self.dataloader = dataloader
        self.class_names = class_names
        self.device = torch.device(device)
        self.half = half & (self.device.type != "cpu")  # half precision only supported on CUDA
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    @torch.no_grad()
    def val_for_one_epoch(self, model):

        logger.debug("start validation ----------------------------------")

        assert hasattr(model, "strides")
        model.half() if self.half else model.float()

        # Configure
        model = model.eval().to(self.device)  # double check
        nc = len(self.class_names)  # number of classes
        iouv = torch.linspace(0.5, 0.95, 10).to(self.device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()

        # Metrics
        seen = 0
        names = {k: v for k, v in enumerate(self.class_names)}
        logger.debug(("%20s" + "%11s" * 6) % ("Class", "Images", "Labels", "P", "R", "mAP@.5", "mAP@.5:.95"))
        p, r, f1, mp, mr, map50, map = [torch.zeros(1) for _ in range(7)]
        stats, ap, ap_class = [], [], []

        for img, targets in self.dataloader:
            img = img.to(self.device, non_blocking=True)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            targets = targets.to(self.device)

            # Run model
            out = model(img)  # inference outputs

            # Run NMS
            out = non_max_suppression(out, self.conf_thres, self.iou_thres)

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
                    correct = val_batch(predn, labels, iouv)
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
        logger.debug(pf % ("all", seen, nt.sum(), mp, mr, map50, map))

        # Print results per class
        if nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                logger.debug(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        logger.debug("------------------------------------------------")

        # Return results
        return mp, mr, map50, map  # P, R, mAP@.5, mAP@.5-.95


class Controller:
    def __init__(self, trainer: Trainer, validator: Validator, patience=16):
        self.trainer = trainer
        self.validator = validator
        self.patience = patience

        # Init wandb directories
        wandb_dir = "./runs/wandb"
        if not os.path.exists(wandb_dir):
            os.makedirs(wandb_dir)
        wandb_logger = wandb.init(project="nano", dir="./runs", mode="offline")
        w = increment_path(Path("runs/train") / "exp", exist_ok=False)
        wandb_logger.config.save_path = str(w)
        w.mkdir(parents=True, exist_ok=True)  # make dir
        self.w = w
        self.wandb_logger = wandb_logger

    def run(self):
        try:
            # Start training
            last, best = self.w / "last.pt", self.w / "best.pt"

            best_fitness = 0
            stopper = EarlyStopping(patience=self.patience)
            for epoch in range(self.trainer.start_epoch, self.trainer.end_epoch):
                mloss = self.trainer.train_for_one_epoch(epoch)
                model = deepcopy(self.trainer.ema.ema)
                metrics = self.validator.val_for_one_epoch(model)
                torch.cuda.empty_cache()

                # Update best mAP
                fi = fitness(np.array(metrics).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                if fi > best_fitness:
                    best_fitness = fi
                logger.info(f"fitness = {fi}, best_fitness = {best_fitness}")

                # Upload logger
                log_vals = {
                    "epoch": epoch,
                    "precision": metrics[0].item(),
                    "recall": metrics[1].item(),
                    "map@.5": metrics[2].item(),
                    "mAP@.5-.95": metrics[3].item(),
                    "train_loss_box": mloss[0].item(),
                    "train_loss_qfl": mloss[1].item(),
                    "train_loss": mloss.sum().item(),
                }
                self.wandb_logger.log(log_vals)

                # Save model
                ckpt = {"state_dict": model.half().state_dict()}
                ckpt.update(log_vals)

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                    logger.info(f"saved ckpt as {best}")
                del ckpt

                # Stop Single-GPU
                if stopper(epoch=epoch, fitness=fi):
                    break
            return best
        except Exception as e:
            logger.error(e)
            raise e
        finally:
            self.wandb_logger.finish()
            torch.cuda.empty_cache()
