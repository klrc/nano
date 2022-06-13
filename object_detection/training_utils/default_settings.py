import torch


class DefaultSettings:
    # model settings
    frozen_dict = None
    grid_stride = 32
    auto_anchor = True

    # optimizer settings
    optimizer = torch.optim.SGD
    input_shape = (4, 3, 360, 640)
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
    lrf = 0.2
    cos_lr = True  # cosine LR scheduler

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
    start_epoch = 0
    max_epoch = 300
    patience = 50  # EarlyStopping patience (epochs without improvement)
    save_dir = "runs/unknown_model"
