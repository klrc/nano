from loguru import logger
import torch
import torch.nn as nn
import numpy as np
import cv2
from multiprocessing import Queue
from test_utils.detection_visualize import Canvas
from test_utils.video_loader import VideoLoader
from test_utils.utils import non_max_suppression, im2tensor
import math
import os


def letterbox_padding(frame, gs=32):
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


def uniform_scale(image, inf_size):
    height, width, _ = image.shape
    ratio = inf_size / max(height, width)
    image = cv2.resize(image, (int(ratio * width), int(ratio * height)))
    return image


def yolov5_inference(model, frame, conf_thres=0.2, iou_thres=0.45):
    with torch.no_grad():
        # Inference
        out, _ = model(frame)  # inference, loss outputs
        # list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        out = non_max_suppression(out, conf_thres, iou_thres, labels=[], multi_label=True, agnostic=True)
    return out


def detect(model: nn.Module, frame: np.ndarray, canvas=None, mode="yolov5", ground_truth=None, class_names=None):
    assert not model.training  # detect in eval mode
    frame = im2tensor(frame).unsqueeze(0)
    frame = letterbox_padding(frame)
    # init canvas
    if canvas is None:
        canvas = Canvas()
    canvas.load(frame[0])
    height, width = frame.shape[-2:]
    # render ground truth
    if ground_truth:
        canvas.load(frame[0])
        for cls, x1, y1, x2, y2 in ground_truth:
            pt1, pt2 = (x1 * width, y1 * height), (x2 * width, y2 * height)
            color = canvas.color(cls)
            title = str(cls) if not class_names else class_names[cls]
            canvas.draw_box(pt1, pt2, alpha=0.4, thickness=-1, color=color)
            canvas.draw_box(pt1, pt2, color=color, title=title)
        canvas.show("ground truth")
    # render prediction
    if mode == "yolov5":
        for det in yolov5_inference(model, frame)[0]:
            pt1, pt2, conf, cls = det[:2], det[2:4], det[4], int(det[5])
            color = canvas.color(cls)
            title = f"{str(cls) if not class_names else class_names[cls]}: {conf:.2f}"
            canvas.draw_box(pt1, pt2, alpha=0.4, thickness=-1, color=color)
            canvas.draw_box(pt1, pt2, color=color, title=title)
        canvas.show("prediction")
    else:
        raise NotImplementedError


def read_labels_as_xyxy(lb):
    if not os.path.exists(lb):
        return []
    # read labels as c, x1, y1, x2, y2
    with open(lb, "r") as f:
        labels = []
        for line in f.readlines():
            c, x, y, w, h = [float(data) for data in line.split(" ")]
            labels.append([int(c), x - w / 2, y - h / 2, x + w / 2, y + h / 2])
        return labels


def travel_dataset(path, class_names=None):
    """
    1~9: jump to n% of dataset
    a: prev image
    *: next image
    q: quit
    """
    # load image paths in queue
    test_queue = []
    for image in os.listdir(path):
        if not image.startswith("._") and (image.endswith(".png") or image.endswith(".jpg")):
            fp = f"{path}/{image}"
            test_queue.append(fp)
    # control loop
    i = 0
    canvas = Canvas()
    while i < len(test_queue):
        fp = test_queue[i]
        print(fp)
        image = cv2.imread(fp)
        lb = fp.replace(".png", ".txt").replace(".jpg", ".txt").replace("/images", "/labels")
        lb = read_labels_as_xyxy(lb)
        image = im2tensor(image).unsqueeze(0)
        height, width = image.shape[-2:]
        canvas.load(image[0])
        for cls, x1, y1, x2, y2 in lb:
            pt1, pt2 = (x1 * width, y1 * height), (x2 * width, y2 * height)
            color = canvas.color(cls)
            title = str(cls) if not class_names else class_names[cls]
            canvas.draw_box(pt1, pt2, alpha=0.4, thickness=-1, color=color)
            canvas.draw_box(pt1, pt2, color=color, title=title)
        canvas.show("ground truth")
        flag = cv2.waitKey(0)
        if flag == ord("q"):
            break
        elif flag == ord("a"):
            i -= 1
        elif flag in [ord(x) for x in "1234567890"]:
            i = int(len(test_queue) * (flag - ord("0")) / 10)
        else:
            i += 1


def full_dataset_test(model, path, class_names, inf_size=640):
    """
    1~9: jump to n% of dataset
    a: prev image
    *: next image
    q: quit
    """
    # load image paths in queue
    test_queue = []
    for image in os.listdir(path):
        if not image.startswith("._") and (image.endswith(".png") or image.endswith(".jpg")):
            fp = f"{path}/{image}"
            test_queue.append(fp)
    # control loop
    i = 0
    canvas = Canvas()
    while i < len(test_queue):
        fp = test_queue[i]
        print(fp)
        image = cv2.imread(fp)
        image = uniform_scale(image, inf_size)
        lb = fp.replace(".png", ".txt").replace(".jpg", ".txt").replace("/images", "/labels")
        lb = read_labels_as_xyxy(lb)
        detect(model, image, canvas, ground_truth=lb, class_names=class_names)
        flag = cv2.waitKey(0)
        if flag == ord("q"):
            break
        elif flag == ord("a"):
            i -= 1
        elif flag in [ord(x) for x in "1234567890"]:
            i = int(len(test_queue) * (flag - ord("0")) / 10)
        else:
            i += 1


def video_test(model, loader: VideoLoader, class_names, fps=24, inf_size=640):
    # run model on a background-process video loader
    # init video loader
    assert isinstance(loader, VideoLoader)
    pipe = Queue(maxsize=2)
    loader.play(pipe, fps=fps)
    # control loop
    canvas = Canvas()
    frame = pipe.get()
    while frame is not None:
        frame = uniform_scale(frame, inf_size)
        detect(model, frame, canvas, class_names=class_names)
        frame = pipe.get()
        # Press ESC to break
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


def camera_test(model, class_names, inf_size=640):
    # inference using camera stream
    camera = cv2.VideoCapture(0)
    assert camera.isOpened()
    size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("size: " + repr(size))
    # warmup
    canvas = Canvas()
    grabbed, frame_lwpCV = camera.read()
    grabbed, frame_lwpCV = camera.read()
    grabbed, frame_lwpCV = camera.read()
    while True:
        # 读取视频流
        grabbed, frame_lwpCV = camera.read()
        if frame_lwpCV is not None:
            frame = uniform_scale(frame_lwpCV, inf_size)
            detect(model, frame, canvas, class_names=class_names)
        # Press ESC to break
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
