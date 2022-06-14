import torch
import torch.nn as nn
import numpy as np
from .utils import non_max_suppression, im2tensor, tensor2im
import math
import cv2


def letterbox_padding(frame, gs=32):
    n, c, w, h = frame.shape
    exp_w = math.ceil(w / gs) * gs
    exp_h = math.ceil(h / gs) * gs
    background = torch.zeros(n, c, exp_w, exp_h, device=frame.device)
    pad_w = (exp_w - w) // 2
    pad_h = (exp_h - h) // 2
    background[:, :, pad_w : pad_w + w, pad_h : pad_h + h] = frame
    return background


def yolov5_inference(model, frame, conf_thres=0.01, iou_thres=0.45):
    with torch.no_grad():
        # Inference
        out, _ = model(frame)  # inference, loss outputs
        # list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        out = non_max_suppression(out, conf_thres, iou_thres, labels=[], multi_label=True, agnostic=False)


def detect(model: nn.Module, frame: np.ndarray, mode="yolov5"):
    assert not model.training  # detect in eval mode
    frame = im2tensor(frame).unsqueeze(0)
    frame = letterbox_padding(frame)
    cv2.imshow("test", tensor2im(frame[0]))
    cv2.waitKey(0)
    if mode == "yolov5":
        nx6 = yolov5_inference(model, frame)
    else:
        raise NotImplementedError
