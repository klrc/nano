from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from object_detection.training_utils.yolov5_autoanchor_pack.autoanchor import check_anchors
from torch.cuda import amp

from .early_stop import EarlyStopping
from .ema import ModelEMA
from .general import (
    check_before_training,
    create_dataloader,
    create_optimizer,
    create_scheduler,
    increment_path,
    initialize_detect_bias,
    save_data,
    select_device,
    sync_settings,
    sync_status,
    sync_yolov5_hyp,
    train_for_one_epoch,
    val_for_one_epoch,
)
from .yolov5_loss_func_pack import ComputeLoss

DEBUG_VALIDATION_ONLY = False
if DEBUG_VALIDATION_ONLY:
    logger.warning("DEBUG_VALIDATION_ONLY mode on!")


def train(model: nn.Module, config=None, device="cpu"):

    s = sync_settings(config)
    device = select_device(device)
    model = check_before_training(model, device, s.input_shape, s.expected_output_shapes)
    optimizer = create_optimizer(model, s.frozen_params, s.optimizer, s.lr0, s.momentum, s.weight_decay)
    scheduler, lf = create_scheduler(optimizer, s.lrf, s.start_epoch, s.max_epoch, s.cos_lr)
    train_loader, trainset = create_dataloader(s.trainset_path, training=True, settings=s)
    val_loader, valset = create_dataloader(s.valset_path, training=False, settings=s)
    if s.auto_anchor:
        check_anchors(valset, model=model, thr=s.anchor_t, imgsz=s.imgsz)
    if s.init_detect_bias:
        initialize_detect_bias(model, trainset, s.nc)

    hyp = sync_yolov5_hyp(model, s)

    ema = ModelEMA(model)
    scaler = amp.GradScaler(enabled=(device.type != "cpu"))
    stopper = EarlyStopping(patience=s.patience)
    criteria = ComputeLoss(model, hyp)

    u = sync_status(train_loader, s)
    w = Path(str(increment_path(s.save_dir, mkdir=True)))
    last, best = w / "last.pt", w / "best.pt"
    save_data(s, w / "settings.txt")

    for epoch in range(s.start_epoch, s.max_epoch):
        u.current_epoch = epoch
        if DEBUG_VALIDATION_ONLY:
            val_for_one_epoch(model, device, val_loader, criteria, s, u)
            break
        else:
            train_for_one_epoch(model, device, train_loader, optimizer, criteria, scaler, ema, scheduler, lf, s, u)
            val_for_one_epoch(ema.ema, device, val_loader, criteria, s, u)

        # save model
        save_data(u, w / "status.txt", "a")
        state_dict = deepcopy(ema.ema).half().state_dict()
        save_data(state_dict, last)
        if u.best_fitness == u.current_fitness:
            save_data(state_dict, best)
            logger.success(f"fitness = {u.best_fitness}, save model as {best}")
        del state_dict

        if stopper(epoch=epoch, fitness=u.current_fitness):
            break

    torch.cuda.empty_cache()
