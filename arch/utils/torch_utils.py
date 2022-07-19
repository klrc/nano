from copy import deepcopy
import math
import os
import torch
import torch.nn as nn
import random
import pkg_resources as pkg
from loguru import logger
import numpy as np


def check_version(current="0.0.0", minimum="0.0.0", name="version ", pinned=False, hard=False, verbose=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f"{name}{minimum} required, but {name}{current} is currently installed"  # string
    if hard:
        assert result, s  # assert min requirements met
    if verbose and not result:
        logger.warning(s)
    return result


def init_seeds(seed=0, deterministic=False):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn

    if deterministic and check_version(torch.__version__, "1.12.0"):  # https://github.com/ultralytics/yolov5/pull/8213
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # for multi GPU, exception safe


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


def create_optimizer(model: nn.Module, optimizer_prototype, lr0, momentum, weight_decay) -> torch.optim.Optimizer:
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


def create_scheduler(optimizer, lrf, final_epoch=300, cos_lr=False):
    if cos_lr:
        lf = one_cycle_lambda(1, lrf, final_epoch)  # cosine 1->hyp['lrf']
    else:
        lf = linear_lambda(lrf, final_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    return scheduler, lf


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model:nn.Module, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

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
        if stop:
            logger.info(
                f"Stopping training early as no improvement observed in last {self.patience} epochs. "
                f"Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n"
                f"To update EarlyStopping(patience={self.patience}) pass a new patience value, "
                f"i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping."
            )
        return stop
