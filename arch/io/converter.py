import torch
import numpy as np
import cv2
from torchvision.transforms.functional import to_tensor


def im2tensor(image: np.ndarray):
    assert np.issubsctype(image, np.integer)  # 0~255 BGR int -> 0~1 RGB float
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return to_tensor(image)


def tensor2im(x: torch.Tensor):
    assert isinstance(x, torch.Tensor) and len(x.shape) == 3  # 0~1 RGB float -> 0~255 BGR int
    np_img = (x * 255).int().numpy().astype(np.uint8)
    np_img = np_img[::-1].transpose((1, 2, 0))  # CHW to HWC, RGB to BGR, 0~1 to 0~255
    np_img = np.ascontiguousarray(np_img)
    return np_img

