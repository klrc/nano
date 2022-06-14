from matplotlib.pyplot import title
import torch
import torch.nn as nn
import numpy as np

from test_utils.detection_visualize import Canvas
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
    return out


def detect(model: nn.Module, frame: np.ndarray, mode="yolov5", class_names=None):
    assert not model.training  # detect in eval mode
    frame = im2tensor(frame).unsqueeze(0)
    frame = letterbox_padding(frame)
    canvas = Canvas(frame[0])
    canvas.show()
    if mode == "yolov5":
        for det in yolov5_inference(model, frame)[0]:
            pt1, pt2, conf, cls = det[:2], det[2:4], det[4], int(det[5])
            color = canvas.color(cls)
            print(cls, conf)
            canvas.draw_box(pt1, pt2, color=color, title=f"{class_names[cls]}:{conf:.2f}")
        canvas.show()
    else:
        raise NotImplementedError
