from typing import Dict
import torch
import torch.nn as nn
import os
import math
from loguru import logger
from tqdm import tqdm
from torch.cuda import amp

from .yolov5_dataset_loader_pack import LoadImagesAndLabels, InfiniteDataLoader
from .yolov5_loss_func_pack import ComputeLoss


class DefaultSettings:
    # model settings
    frozen_dict = None
    grid_stride = 32

    # optimizer settings
    optimizer = torch.optim.SGD
    input_shape = (4, 3, 360, 640)
    imgsz = 640
    expected_output_shapes = None
    lr0 = 0.01
    momentum = 0.937
    weight_decay = 0.0005

    # scheduler settings
    warmup_epochs = 3
    warmup_momentum = 0.8
    warmup_bias_lr = 0.1
    lrf = 0.01
    cos_lr = False

    # dataloader settings
    trainset_path = ...
    valset_path = ...
    batch_size = 64
    cache = False
    workers = 8

    # augmentation settings
    mosaic = 1.0
    mixup = 0.0
    degrees = 0.0
    translate = 0.1
    scale = 0.5
    shear = 0.0
    perspective = 0.0
    hsv_h = 0.015
    hsv_s = 0.7
    hsv_v = 0.4
    flipud = 0.0
    fliplr = 0.5
    copy_paste = 0.0

    # other settings
    start_epoch = 0
    max_epoch = 300
    save_path = ...


# functions def =================================================================================


def sync_settings(config: Dict):
    s = DefaultSettings()
    if config is not None:
        for k, v in config.items():
            if hasattr(s, k):
                setattr(s, k, v)
    return s


def select_device(device="", batch_size=0, newline=True):
    # device = 'cpu' or '0' or '0,1,2,3'
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
    except Exception as e:
        logger.error("model inference check failed")
        raise e


def create_optimizer(model: nn.Module, frozen_layers, optimizer_prototype, lr0, momentum, weight_decay):
    for k, v in model.named_parameters():
        v.requires_grad = True
        if frozen_layers is not None and any(x in k for x in frozen_layers):
            v.requires_grad = False

    pgroup = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):  # bias
            pgroup[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            pgroup[1].append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            pgroup[0].append(v.weight)

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
    return optimizer


def one_cycle_lambda(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def linear_lambda(lrf, epochs):
    return lambda x: (1 - x / epochs) * (1.0 - lrf) + lrf  # linear


def create_scheduler(optimizer, lrf, epochs, cos_lr=False):
    # Scheduler
    if cos_lr:
        lf = one_cycle_lambda(1, lrf, epochs)  # cosine 1->hyp['lrf']
    else:
        lf = linear_lambda(lrf, epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    return scheduler


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
        augment, shuffle, rect, pad = False, False, True, 0.5
        batch_size *= 2
        workers *= 2
    dataset = LoadImagesAndLabels(
        dataset_path, s.imgsz, batch_size, augment=augment, hyp=hyp, rect=rect, cache_images=s.cache, stride=int(s.grid_stride), pad=pad
    )
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    loader = InfiniteDataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=nw, sampler=None, pin_memory=True, collate_fn=LoadImagesAndLabels.collate_fn
    )
    return loader


def inference(model, batch_data, criteria, training, device):
    if training:
        model.train()
        imgs, targets, paths, _ = batch_data
        imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
        cuda = device.type != "cpu"
        with amp.autocast(enabled=cuda):
            pred = model(imgs)  # forward
            loss, loss_items = criteria(pred, targets.to(device))  # loss scaled by batch_size
        return loss, loss_items
    else:
        model.eval()
        return output, target



def train(model: nn.Module, config=None, device="cpu"):
    # on training start
    s = sync_settings(config)
    device = select_device(device)
    model = check_before_training(model, device, s.input_shape, s.expected_output_shapes)
    optimizer = create_optimizer(model, s.frozen_dict, s.optimizer, s.lr0, s.momentum, s.weight_decay)
    scheduler = create_scheduler(optimizer, s.lrf, s.max_epoch, s.cos_lr)
    scaler = amp.GradScaler(enabled=(device.type != "cpu"))
    train_loader = create_dataloader(s.trainset_path, training=True, settings=s)
    val_loader = create_dataloader(s.valset_path, training=False, settings=s)
    criteria = ComputeLoss(model)

    on_training_start()
    for epoch in range(s.start_epoch, s.max_epoch):
        # training pipeline
        nb = len(train_loader)  # number of batches
        mloss = torch.zeros(3, device=device)  # mean losses
        pbar = enumerate(train_loader)
        print(("\n" + "%10s" * 7) % ("Epoch", "gpu_mem", "box", "obj", "cls", "labels", "img_size"))
        pbar = tqdm(pbar, total=nb, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")  # progress bar
        model.train()
        optimizer.zero_grad()
        for i, batch_data in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            loss, loss_items = inference(model, batch_data, criteria, training=True, device=device)
            scaler.scale(loss).backward()
            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni


        # validation pipeline
        model.eval()
        for batch_data in valoader:
            output, target = inference(model, batch_data, training=False)
            metrics = benchmark_test(output, target)
        early_stop = on_training_epoch_end(metrics, model, s.save_path)
        if early_stop:
            break
    # on training finish
    on_training_finish()
