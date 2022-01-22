import numpy as np
import torch


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


import numpy as np
import torch
import cv2


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


def draw_center_point(image, center, color=None, thickness=3, alpha=None):
    """
    draw square pixel-style points on image
    """
    # set image format
    canvas = any_image_format(image)  # CHW BGR 0~255
    # set tensor format (to numpy)
    center = any_tensor_format(center)
    center = [int(x) for x in center]  # to int
    # set color
    color = rand_default_color(color)
    # draw point
    if alpha is None:
        cv2.circle(canvas, center, 1, color, thickness)
    else:
        alpha = float(alpha)
        p = cv2.circle(canvas.copy(), center, 1, color, thickness)
        canvas = cv2.addWeighted(canvas, 1 - alpha, p, alpha, 0)
    return canvas


def draw_bounding_box(image, box, color=None, alpha=None):
    # set image format
    canvas = any_image_format(image)  # CHW BGR 0~255
    # set color
    color = rand_default_color(color)
    # draw box
    x1, y1, x2, y2 = [int(x) for x in box]
    if alpha is None:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1, 4, 0)
    else:
        alpha = float(alpha)
        p = cv2.rectangle(canvas.copy(), (x1, y1), (x2, y2), color, 1, 4, 0)
        canvas = cv2.addWeighted(canvas, 1 - alpha, p, alpha, 0)
    return canvas


def draw_text_with_background(
    image, text, x1, y1, color=None, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, font_thickness=1, font_color=(0, 0, 0), alpha=None
):
    """
    draw labels with auto-fitting background color
    """
    # set image format
    canvas = any_image_format(image)  # CHW BGR 0~255
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    if alpha is None:
        cv2.rectangle(canvas, (x1, y1), (x1 + text_w + 2, y1 + text_h + 2), color, -1)
        cv2.putText(canvas, text, (x1, y1 + text_h), font, font_scale, font_color, font_thickness)
    else:
        alpha = float(alpha)
        p = cv2.rectangle(canvas.copy(), (x1, y1), (x1 + text_w + 2, y1 + text_h + 2), color, -1)
        cv2.putText(p, text, (x1, y1 + text_h), font, font_scale, font_color, font_thickness)
        canvas = cv2.addWeighted(canvas, 1 - alpha, p, alpha, 0)
    return canvas


def draw_center_points(image, centers, color=None, thickness=3, alphas=None):
    # set image format
    image = any_image_format(image)  # CHW BGR 0~255
    # set tensor format (to numpy)
    centers = any_tensor_format(centers)
    # set color
    color = rand_default_color(color)
    # set alphas
    if alphas is None:
        alphas = [None for _ in range(len(centers))]
    # draw points
    for center, alpha in zip(centers, alphas):
        image = draw_center_point(image, center, color, thickness, alpha)
    return image


def draw_bounding_boxes(image, boxes, box_color=None, boxes_label=None, boxes_centers=None, alphas=None):
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
    if alphas is not None:
        assert len(alphas) == len(boxes)
    # draw boxes
    for i, box in enumerate(boxes):
        # set color
        color = rand_default_color(box_color)
        # set alpha
        alpha = None if alphas is None else alphas[i]
        # draw box
        cv2_img = draw_bounding_box(cv2_img, box, color, alpha)
        # draw label
        if boxes_label is not None:
            x1, y1, _, _ = [int(x) for x in box]
            text = boxes_label[i]
            cv2_img = draw_text_with_background(cv2_img, text, x1, y1, color, alpha=alpha)
        # draw center
        if boxes_centers is not None:
            centers = boxes_centers[i]
            if len(centers.shape) == 1:
                centers = [centers]
            for center in centers:
                cv2_img = draw_center_point(cv2_img, center, color, alpha=alpha)
    return cv2_img


coco_classes = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]  # class names


voc_classes = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]  # class names


c26_classes = [
    # human --------------------
    "person",
    # vehicles -----------------
    "bike",
    "car",
    "plane",
    "train",
    "boat",
    # outdoor ------------------
    "sign",
    "facility",
    # objects -------------------
    "suitcase",
    "projectile",
    "kite",
    # equipments ---------------
    "umbrella",
    "board",
    "equipment",
    # animals ------------------
    "bird",
    "mammal",
    # plants -------------------
    "plant",
    "fruit",
    # indoor -------------------
    "furniture",
    "bottle",
    "food",
    "device",
    "appliances",
    "handtool",
    "clock",
    "toy",
]


# _coco_to_c26 = {
#     "person": "person",
#     "bicycle": "bike",
#     "car": "car",
#     "motorcycle": "bike",
#     "airplane": "plane",
#     "bus": "car",
#     "train": "train",
#     "truck": "car",
#     "boat": "boat",
#     "traffic light": "sign",
#     "fire hydrant": "facility",
#     "stop sign": "sign",
#     "parking meter": "facility",
#     "bench": "furniture",
#     "bird": "bird",
#     "cat": "mammal",
#     "dog": "mammal",
#     "horse": "mammal",
#     "sheep": "mammal",
#     "cow": "mammal",
#     "elephant": "mammal",
#     "bear": "mammal",
#     "zebra": "mammal",
#     "giraffe": "mammal",
#     "umbrella": "umbrella",
#     "suitcase": "suitcase",
#     "frisbee": "projectile",
#     "skis": "board",
#     "snowboard": "board",
#     "sports ball": "projectile",
#     "kite": "kite",
#     "baseball bat": "equipment",
#     "skateboard": "board",
#     "surfboard": "board",
#     "tennis racket": "equipment",
#     "bottle": "bottle",
#     "wine glass": "bottle",
#     "cup": "bottle",
#     "fork": "handtool",
#     "knife": "handtool",
#     "spoon": "handtool",
#     "bowl": "bottle",
#     "banana": "fruit",
#     "apple": "fruit",
#     "sandwich": "food",
#     "orange": "fruit",
#     "broccoli": "fruit",
#     "carrot": "fruit",
#     "hot dog": "food",
#     "pizza": "food",
#     "donut": "food",
#     "cake": "food",
#     "chair": "furniture",
#     "couch": "furniture",
#     "potted plant": "plant",
#     "bed": "furniture",
#     "dining table": "furniture",
#     "toilet": "facility",
#     "tv": "device",
#     "laptop": "device",
#     "mouse": "device",
#     "remote": "device",
#     "keyboard": "device",
#     "cell phone": "device",
#     "microwave": "appliances",
#     "oven": "appliances",
#     "toaster": "appliances",
#     "refrigerator": "appliances",
#     "clock": "clock",
#     "vase": "bottle",
#     "scissors": "handtool",
#     "teddy bear": "toy",
#     "hair drier": "device",
#     "toothbrush": "handtool",
# }


# _voc_to_c26 = {
#     "aeroplane": "plane",
#     "bicycle": "bike",
#     "bird": "bird",
#     "boat": "boat",
#     "bottle": "bottle",
#     "bus": "car",
#     "car": "car",
#     "cat": "mammal",
#     "chair": "furniture",
#     "cow": "mammal",
#     "diningtable": "furniture",
#     "dog": "mammal",
#     "horse": "mammal",
#     "motorbike": "bike",
#     "person": "person",
#     "pottedplant": "plant",
#     "sheep": "mammal",
#     "sofa": "furniture",
#     "train": "train",
#     "tvmonitor": "device",
# }

# _coco_to_c3 = {
#     "person": "person",
#     "bicycle": "bike",
#     "car": "car",
#     "motorcycle": "bike",
#     "bus": "car",
#     "truck": "car",
# }
