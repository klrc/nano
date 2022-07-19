import numpy as np
import math
import cv2
import torch


def letterbox_padding(frame: torch.Tensor, gs=32):
    n, c, w, h = frame.shape
    if w % gs == 0 and h % gs == 0:
        return frame
    exp_w = math.ceil(w / gs) * gs
    exp_h = math.ceil(h / gs) * gs
    background = torch.zeros(n, c, exp_w, exp_h, device=frame.device)
    pad_w = (exp_w - w) // 2
    pad_h = (exp_h - h) // 2
    background[:, :, pad_w : pad_w + w, pad_h : pad_h + h] = frame
    return background


def uniform_scale(image: np.ndarray, inf_size: int):
    height, width, _ = image.shape
    ratio = inf_size / max(height, width)
    image = cv2.resize(image, (int(ratio * width), int(ratio * height)))
    return image


