from copy import deepcopy
import inspect
import numpy as np
from pathlib import Path
import torch.nn as nn
import torch
from torch.cuda import amp
from tqdm import tqdm

from ...data.detection_annotation import coco_meta_data

from ..detection_validation_41c import detection_validation

from ...utils.file_utils import increment_path
from ...utils.torch_utils import EarlyStopping, ModelEMA, create_optimizer, create_scheduler, select_device, init_seeds


class DetectionTraining4C:
    def __init__(self) -> None:
        # Basic settings
        # utility settings
        self.patience = 30  # EarlyStopping patience (epochs without improvement)
        self.save_dir = "runs/untitled_model"
        self.seed = 0
        self.device = "cpu"
        self.debug_mode = False
        # dataset settings
        self.names = coco_meta_data.names  # class names
        self.nc = len(self.names)  # num of classes
        # optimizer settings
        self.optimizer = torch.optim.SGD
        self.input_shape = (4, 3, 384, 640)
        self.imgsz = 640
        self.expected_output_shapes = None
        self.lr0 = 0.01
        self.momentum = 0.937
        self.weight_decay = 0.0005
        self.nbs = 64  # nominal batch size
        # scheduler settings
        self.warmup_epochs = 3
        self.warmup_momentum = 0.8
        self.warmup_bias_lr = 0.1
        self.lrf = 0.01
        self.cos_lr = False  # cosine LR scheduler
        self.final_epoch = 300
        # Dynamic parameters
        self.model_state_dict = None
        self.optimizer_state_dict = None
        self.ema_state_dict = None
        self.ema_updates = None
        self.w = None
        self.stopper = None
        self.best_fitness = 0.0
        self.last_epoch = -1

    def freeze(self, path):
        attrs = {}
        for k, v in inspect.getmembers(self):
            if k.startswith("__"):
                continue
            if k in ("freeze", "resume", "train", "fitness"):
                continue
            attrs[k] = v
        torch.save(attrs, path)
        return self

    @staticmethod
    def fitness(x):
        # Model fitness as a weighted combination of metrics
        w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        return float((x[:, :4] * w).sum(1))

    @staticmethod
    def resume(cache_file, model, train_loader, val_loader, loss_func):
        # loaders are not serializable, so we need to set manually.
        DT = DetectionTraining4C()
        attrs = torch.load(cache_file)
        for k, v in attrs.items():
            setattr(DT, k, v)
        return DT.train(model, train_loader, val_loader, loss_func)

    def train(self, model, train_loader, val_loader, loss_func):

        # Init Seeds
        init_seeds(self.seed, deterministic=True)

        # Check device
        device = select_device(self.device)

        # Check Model
        assert isinstance(model, nn.Module)
        assert hasattr(model, "head")
        assert hasattr(model.head, "anchors")
        assert hasattr(model.head, "stride")
        if self.model_state_dict:
            model.load_state_dict(self.model_state_dict)
        model.to(device)
        anchors = model.head.anchors
        stride = model.head.stride

        # Check Dataloader
        for loader in (train_loader, val_loader):
            assert hasattr(loader, "batch_size")
            assert hasattr(loader, "__len__")
        batch_size = train_loader.batch_size
        nb = len(train_loader)

        # Directories
        if self.w is None:
            self.w = str(Path(str(increment_path(self.save_dir, mkdir=True))))
        path_checkpoint = self.w + "/frozen.core"
        path_best_model = self.w + "/best.pt"

        # Optimizer
        accumulate = max(round(self.nbs / batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.weight_decay * accumulate / self.nbs  # scale weight_decay
        optimizer = create_optimizer(model, self.optimizer, self.lr0, self.momentum, weight_decay)
        if self.optimizer_state_dict:
            optimizer.load_state_dict(self.optimizer_state_dict)

        # Scheduler
        scheduler, lf = create_scheduler(optimizer, self.lrf, self.final_epoch, self.cos_lr)

        # EMA
        ema = ModelEMA(model)
        if self.ema_state_dict:
            ema.ema.load_state_dict(self.ema_state_dict)
            ema.updates = self.ema_updates

        # Auto-anchor
        if hasattr(train_loader, "auto_anchor"):
            # from ...model.detection_utils.head_41c import DH41C
            # from ...data.detection_annotation.yolov5 import Yolo5Dataloader
            # head: DH41C
            # train_loader: Yolo5Dataloader
            model.head.anchors = train_loader.auto_anchor(anchors, stride).to(device)
        model.half().float()  # pre-reduce anchor precision

        # GradScaler
        grad_scaler = amp.GradScaler(enabled=(device.type != "cpu"))

        # Stopper
        if self.stopper is None:
            self.stopper = EarlyStopping(patience=self.patience)

        # Start training
        last_opt_step = -1
        nw = max(round(self.warmup_epochs * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
        for epoch in range(self.last_epoch + 1, self.final_epoch):
            model.train()
            mloss = torch.zeros(2, device=device)  # mean losses
            optimizer.zero_grad()

            # Set progress bar
            pbar = enumerate(train_loader)
            print(("\n" + "%10s" * 6) % ("Epoch", "gpu_mem", "box", "cls", "labels", "img_size"))
            pbar = tqdm(pbar, total=nb, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")  # progress bar

            # Batch loop
            for i, (imgs, targets) in pbar:
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device)

                # Warmup
                ni = i + nb * epoch  # number integrated batches (since train start)
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                    accumulate = max(1, np.interp(ni, xi, [1, self.nbs / batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(ni, xi, [self.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.warmup_momentum, self.momentum])

                # Forward
                with torch.cuda.amp.autocast(True):
                    pred = model(imgs)  # forward
                    loss, loss_items = loss_func(pred, targets)  # loss scaled by batch_size

                # Backward
                grad_scaler.scale(loss).backward()

                # Optimize
                if ni - last_opt_step >= accumulate:
                    grad_scaler.step(optimizer)  # optimizer.step
                    grad_scaler.update()
                    optimizer.zero_grad()
                    ema.update(model)
                    last_opt_step = ni

                # Log
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                pbar.set_description(("%10s" * 2 + "%10.4g" * 4) % (f"{epoch}/{self.final_epoch - 1}", mem, *mloss, targets.shape[0], imgs.shape[-1]))

                # Debug mode
                if self.debug_mode:
                    break

            # Scheduler
            scheduler.step()

            # Update best mAP
            # ema.update_attr(model)
            results, _ = detection_validation(ema.ema, val_loader, self.names, loss_func, device)
            fi = self.fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = self.stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > self.best_fitness:
                self.best_fitness = fi

            # Save model
            self.last_epoch = epoch
            self.model_state_dict = model.state_dict()
            self.optimizer_state_dict = optimizer.state_dict()
            self.ema_state_dict = deepcopy(ema.ema).half().state_dict()
            self.ema_updates = ema.updates
            self.freeze(path_checkpoint)

            # Save last, best and delete
            if self.best_fitness == fi:
                torch.save(self.ema_state_dict, path_best_model)
            del self.model_state_dict, self.optimizer_state_dict, self.ema_state_dict, self.ema_updates

            if stop:
                break
