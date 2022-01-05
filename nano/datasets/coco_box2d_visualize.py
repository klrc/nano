import numpy as np
import torch
import cv2


def from_tensor_image(x):
    """
    converts tensor to numpy image
    returns:
        np_img: (h, w, bgr)
    """
    np_img = (x * 255.0).int().numpy()
    np_img = np_img[::-1].transpose((1, 2, 0))  # CHW to HWC, RGB to BGR, 0~1 to 0~255
    np_img = np.ascontiguousarray(np_img)
    return np_img


def from_numpy_image(img):
    """
    converts numpy image to tensor
    (inplace-safe function)
    returns:
        x: (rgb, h, w)
    """
    x = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    x = np.ascontiguousarray(x)
    x = torch.from_numpy(x).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
    return x


def any_image_format(image):
    if isinstance(image, torch.Tensor):
        return from_tensor_image(image)
    return image


def any_tensor_format(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    return tensor


def rand_default_color(color):
    if color is None:
        return list(np.random.random(size=3) * 128 + 128)  # light color
    return color


def draw_center_points(image, centers, color=None, thickness=3, alphas=None):
    """
    draw square pixel-style points on image
    """
    # set image format
    cv2_img = any_image_format(image)  # CHW BGR 0~255
    # set tensor format (to numpy)
    centers = any_tensor_format(centers)
    # set color
    color = rand_default_color(color)
    # draw points
    assert centers.shape[-1] == 2
    if len(centers.shape) == 1:
        centers = [centers]
    if alphas is not None:
        assert len(alphas) == len(centers)
    for i, cp in enumerate(centers):
        grid = [int(x) for x in cp]
        if alphas is not None:
            alpha = alphas[i]
            point_color = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
        else:
            point_color = color
        cv2_img = cv2.circle(cv2_img, grid, 1, point_color, thickness=thickness)

    return cv2_img


def draw_text_with_background(image, text, x1, y1, color=None, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, font_thickness=1, font_color=(0, 0, 0)):
    """
    draw labels with auto-fitting background color
    """
    # set image format
    cv2_img = any_image_format(image)  # CHW BGR 0~255
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(cv2_img, (x1, y1), (x1 + text_w + 2, y1 + text_h + 2), color, -1)
    cv2.putText(cv2_img, text, (x1, y1 + text_h), font, font_scale, font_color, font_thickness)
    return cv2_img


def draw_bounding_boxes(image, boxes, box_color=None, boxes_label=None, boxes_centers=None):
    """
    draw bounding boxes on image
    boxes: x1, y1, x2, y2
    boxes_label: list(str)  string labels of each box
    boxes_centers: list(centers)  each center(s) should be tensor/numpy array
    if no color is set, random colors will be applied
    """
    # set image format
    cv2_img = any_image_format(image)  # CHW BGR 0~255
    if boxes_label is not None:
        assert len(boxes_label) == len(boxes)
    if boxes_centers is not None:
        assert len(boxes_centers) == len(boxes)
    # draw boxes
    for i, box in enumerate(boxes):
        # set color
        color = rand_default_color(box_color)
        # draw box
        x1, y1, x2, y2 = [int(x) for x in box]
        cv2.rectangle(cv2_img, (x1, y1), (x2, y2), color, 1, 4, 0)
        # draw label
        if boxes_label is not None:
            text = boxes_label[i]
            cv2_img = draw_text_with_background(cv2_img, text, x1, y1, color)
        # draw center
        if boxes_centers is not None:
            cv2_img = draw_center_points(cv2_img, boxes_centers[i], color=color)
    return cv2_img
