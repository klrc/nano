from copy import deepcopy
from typing import Iterable
from loguru import logger
import torch
from torch.cuda import amp
from pathlib import Path


from .utils.sample_settings import SampleSettings
from .utils.yolov5_autoanchor_pack.autoanchor import check_anchors

from .utils.early_stop import EarlyStopping
from .utils.yolov5_loss_func_pack.loss import ComputeLoss
from .utils.ema import ModelEMA
from .utils.general import (
    check_forward,
    create_dataloader,
    create_optimizer,
    create_scheduler,
    increment_path,
    initialize_detect_bias,
    save_data,
    select_device,
    sync_status,
    sync_yolov5_hyp,
    multi_task_backward,
    val_for_one_epoch,
)


class PresetScratch(SampleSettings):
    # model settings
    frozen_params = None
    grid_stride = 32
    auto_anchor = True
    init_detect_bias = True

    # optimizer settings
    optimizer = torch.optim.SGD
    input_shape = (4, 3, 384, 640)
    imgsz = 640
    expected_output_shapes = None
    lr0 = 0.01
    momentum = 0.937
    weight_decay = 0.0005
    nbs = 64  # nominal batch size

    # scheduler settings
    warmup_epochs = 3
    warmup_momentum = 0.8
    warmup_bias_lr = 0.1
    lrf = 0.01
    cos_lr = False  # cosine LR scheduler

    # dataset settings
    nc = 80  # num of classes
    names = "person|bicycle|car|motorcycle|airplane|bus|train|truck|boat|traffic light|fire hydrant|stop sign|parking meter|bench|bird|cat|dog|horse|sheep|cow|elephant|bear|zebra|giraffe|backpack|umbrella|handbag|tie|suitcase|frisbee|skis|snowboard|sports ball|kite|baseball bat|baseball glove|skateboard|surfboard|tennis racket|bottle|wine glass|cup|fork|knife|spoon|bowl|banana|apple|sandwich|orange|broccoli|carrot|hot dog|pizza|donut|cake|chair|couch|potted plant|bed|dining table|toilet|tv|laptop|mouse|remote|keyboard|cell phone|microwave|oven|toaster|sink|refrigerator|book|clock|vase|scissors|teddy bear|hair drier|toothbrush".split(  # noqa:E501
        "|"
    )  # class names

    # dataloader settings
    trainset_path = "../datasets/coco128/images/train2017"
    valset_path = "../datasets/coco128/images/train2017"
    batch_size = 64
    cache = False  # cache images in "ram" (default) or "disk"
    workers = 8  # max dataloader workers

    # augmentation settings
    mosaic = 1.0  # image mosaic (probability)
    mixup = 0.0  # image mixup (probability)
    degrees = 0.0  # image rotation (+/- deg)
    translate = 0.1  # image translation (+/- fraction)
    scale = 0.5  # image scale (+/- gain)
    shear = 0.0  # image shear (+/- deg)
    perspective = 0.0  # image perspective (+/- fraction), range 0-0.001
    hsv_h = 0.015
    hsv_s = 0.7
    hsv_v = 0.4
    flipud = 0.0  # image flip up-down (probability)
    fliplr = 0.5  # image flip left-right (probability)
    copy_paste = 0.0
    fake_osd = False
    fake_darkness = False

    # training settings
    label_smoothing = 0.0  # label smoothing epsilon
    cls_pw = 1.0  # cls BCELoss positive_weight
    obj_pw = 1.0  # obj BCELoss positive_weight
    box = 0.05  # box loss gain
    cls = 0.5  # cls loss gain
    obj = 1  # obj loss gain (scale with pixels)
    anchor_t = 4.0  # anchor-multiple threshold
    fl_gamma = 0.0  # focal loss gamma (efficientDet default gamma=1.5)

    # validation settings
    half = False  # use FP16 half-precision inference
    compute_loss = True
    conf_thres = 0.001  # confidence threshold
    iou_thres = 0.6  # NMS IoU threshold
    verbose = True  # report mAP by class

    # other settings
    save_plot = True
    start_epoch = 0
    max_epoch = 300
    patience = 30  # EarlyStopping patience (epochs without improvement)
    save_dir = "runs/unknown_model"


class PresetFineTuning(SampleSettings):
    # ==============================================================
    # fine-tune (more data & strong augmentation)
    # model settings
    frozen_params = None
    grid_stride = 32
    auto_anchor = True
    init_detect_bias = True

    # optimizer settings
    optimizer = torch.optim.SGD
    input_shape = (4, 3, 384, 640)
    imgsz = 640
    expected_output_shapes = None
    lr0 = 0.003
    momentum = 0.75
    weight_decay = 0.00025
    nbs = 64  # nominal batch size

    # scheduler settings
    warmup_epochs = 3
    warmup_momentum = 0.8
    warmup_bias_lr = 0.1
    lrf = 0.15
    cos_lr = False  # cosine LR scheduler

    # dataset settings
    nc = 80  # num of classes
    names = "person|bicycle|car|motorcycle|airplane|bus|train|truck|boat|traffic light|fire hydrant|stop sign|parking meter|bench|bird|cat|dog|horse|sheep|cow|elephant|bear|zebra|giraffe|backpack|umbrella|handbag|tie|suitcase|frisbee|skis|snowboard|sports ball|kite|baseball bat|baseball glove|skateboard|surfboard|tennis racket|bottle|wine glass|cup|fork|knife|spoon|bowl|banana|apple|sandwich|orange|broccoli|carrot|hot dog|pizza|donut|cake|chair|couch|potted plant|bed|dining table|toilet|tv|laptop|mouse|remote|keyboard|cell phone|microwave|oven|toaster|sink|refrigerator|book|clock|vase|scissors|teddy bear|hair drier|toothbrush".split(  # noqa:E501
        "|"
    )  # class names

    # dataloader settings
    trainset_path = "../datasets/coco128/images/train2017"
    valset_path = "../datasets/coco128/images/train2017"
    batch_size = 64
    cache = False  # cache images in "ram" (default) or "disk"
    workers = 8  # max dataloader workers

    # augmentation settings
    mosaic = 0.9  # image mosaic (probability)
    mixup = 0.04  # image mixup (probability)
    degrees = 0.0  # image rotation (+/- deg)
    translate = 0.1  # image translation (+/- fraction)
    scale = 0.75  # image scale (+/- gain)
    shear = 0.0  # image shear (+/- deg)
    perspective = 0.0  # image perspective (+/- fraction), range 0-0.001
    hsv_h = 0.015
    hsv_s = 0.7
    hsv_v = 0.4
    flipud = 0.0  # image flip up-down (probability)
    fliplr = 0.5  # image flip left-right (probability)
    copy_paste = 0.0
    fake_osd = True
    fake_darkness = False

    # training settings
    label_smoothing = 0.0  # label smoothing epsilon
    cls_pw = 1.0  # cls BCELoss positive_weight
    obj_pw = 1.0  # obj BCELoss positive_weight
    box = 0.05  # box loss gain
    cls = 0.5  # cls loss gain
    obj = 1  # obj loss gain (scale with pixels)
    anchor_t = 4.0  # anchor-multiple threshold
    fl_gamma = 0.0  # focal loss gamma (efficientDet default gamma=1.5)

    # validation settings
    half = False  # use FP16 half-precision inference
    compute_loss = True
    conf_thres = 0.001  # confidence threshold
    iou_thres = 0.6  # NMS IoU threshold
    verbose = True  # report mAP by class

    # other settings
    save_plot = True
    start_epoch = 0
    max_epoch = 100
    patience = 10  # EarlyStopping patience (epochs without improvement)
    save_dir = "runs/unknown_model"


class MultiTaskControlUnit:
    def __init__(self, model, task_settings: Iterable[SampleSettings], task_weights, task_ports, main_task=0):
        self.model = model
        self.tasks = [
            {
                "settings": s,
                "weight": w,
                "ports": p,
                "is_main_task": False,
            }
            for s, w, p in zip(task_settings, task_weights, task_ports)
        ]
        self.tasks[main_task]["is_main_task"] = True
        self.DEBUG_VALIDATION_ONLY = False
        self.FAST_DEBUG_MODE = False

    def load_settings(self) -> SampleSettings:
        # no default settings allowed in mcu
        main_task = None
        for task in self.tasks:
            s = task["settings"]
            assert isinstance(s, SampleSettings)
            if task["is_main_task"]:
                main_task = task

        # move main_task to task head
        assert main_task is not None
        return main_task["settings"]

    def check_all_modules(self):
        # model check
        model = self.model
        assert isinstance(model, torch.nn.Module)
        assert hasattr(model, "detect")
        assert hasattr(model, "fuse")
        assert hasattr(model.detect, "nc")
        assert hasattr(model.detect, "no")
        assert hasattr(model.detect, "nl")
        assert hasattr(model.detect, "na")
        assert hasattr(model.detect, "anchors")
        assert hasattr(model.detect, "stride")

        # io & data module is capsulated inside this unit
        pass

    def run(self, device="cpu"):
        # check if all modules are available
        self.check_all_modules()
        if self.DEBUG_VALIDATION_ONLY:
            logger.warning("DEBUG_VALIDATION_ONLY mode on!")
        if self.FAST_DEBUG_MODE:
            logger.warning("FAST_DEBUG_MODE mode on!")

        # initialize training environment
        s = self.load_settings()
        device = select_device(device)
        model = check_forward(self.model, device, s.input_shape, s.expected_output_shapes)
        optimizer = create_optimizer(model, s.frozen_params, s.optimizer, s.lr0, s.momentum, s.weight_decay)
        scheduler, lf = create_scheduler(optimizer, s.lrf, s.start_epoch, s.max_epoch, s.cos_lr)

        # dont have to sync dataloader batchsize
        for i, task in enumerate(self.tasks):
            ts = task["settings"]
            train_loader, trainset = create_dataloader(ts.trainset_path, training=True, settings=ts)
            val_loader, valset = create_dataloader(ts.valset_path, training=False, settings=ts)
            if task["is_main_task"]:
                if ts.auto_anchor:
                    check_anchors(valset, model=model, thr=ts.anchor_t, imgsz=ts.imgsz)
                if ts.init_detect_bias:
                    initialize_detect_bias(model, trainset, ts.nc, device)

            # yolov5 hyper-parameter scaling
            hyp = sync_yolov5_hyp(model, ts)

            # register tasks
            self.tasks[i]["criteria"] = ComputeLoss(model, hyp)
            self.tasks[i]["train_loader"] = train_loader
            self.tasks[i]["val_loader"] = val_loader

        # initialize training controller
        ema = ModelEMA(model)
        scaler = amp.GradScaler(enabled=(device.type != "cpu"))
        stopper = EarlyStopping(patience=s.patience)

        # initialize output path
        u = sync_status(train_loader, s)
        w = Path(str(increment_path(s.save_dir, mkdir=True)))
        u.save_dir = str(w)
        last, best = w / "last.pt", w / "best.pt"
        save_data(s, w / "settings.txt")

        # start training loop
        for epoch in range(s.start_epoch, s.max_epoch):
            u.current_epoch = epoch
            # train & val for one epoch
            if self.DEBUG_VALIDATION_ONLY:
                for task in self.tasks:
                    task_ports = task["ports"]
                    val_loader = task["val_loader"]
                    is_main_task = task["is_main_task"]
                    val_for_one_epoch(model, task_ports, device, val_loader, s, u, fast_debug_mode=self.FAST_DEBUG_MODE, is_main_task=is_main_task)
                break
            else:
                multi_task_backward(model, device, self.tasks, optimizer, scaler, ema, scheduler, lf, s, u, fast_debug_mode=self.FAST_DEBUG_MODE)
                for task in self.tasks:
                    task_ports = task["ports"]
                    val_loader = task["val_loader"]
                    is_main_task = task["is_main_task"]
                    val_for_one_epoch(ema.ema, task_ports, device, val_loader, s, u, fast_debug_mode=self.FAST_DEBUG_MODE, is_main_task=is_main_task)

            # save model each epoch
            save_data(u, w / "status.txt", "a")
            state_dict = deepcopy(ema.ema).half().state_dict()
            save_data(state_dict, last)
            if u.best_fitness == u.current_fitness:
                save_data(state_dict, best)
                logger.success(f"fitness = {u.best_fitness}, save model as {best}")
            del state_dict

            # early-stop control
            if stopper(epoch=epoch, fitness=u.current_fitness):
                break

        # cleaning cache after training
        self.close()

    def close(self):
        torch.cuda.empty_cache()
