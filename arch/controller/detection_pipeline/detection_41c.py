from copy import deepcopy
import os
from typing import Iterable
import torch.nn as nn
import torch
import numpy as np
import cv2

from ...utils.image_rescale import letterbox_padding, uniform_scale
from ...utils.non_max_suppression import non_max_suppression_41c
from ...ia import Canvas, im2tensor


class DetectPipeline41C:
    def __init__(self, model, class_names=None) -> None:
        self.model = model
        self.class_names = class_names
        self.canvas_title = "prediction"

    @staticmethod
    def detect(canvas_title, model: nn.Module, frame: torch.Tensor, canvas: Canvas, conf_thres, iou_thres, class_names=None):
        if isinstance(frame, np.ndarray):
            frame = im2tensor(frame)
        if len(frame.shape) == 3:
            frame = frame.unsqueeze(0)
        frame = letterbox_padding(frame)
        # init canvas
        canvas.load(frame[0])
        # render prediction
        with torch.no_grad():
            # Inference
            out, _ = model(frame)  # inference, loss outputs
            # list of detections, on (n,6) tensor per image [xyxy, conf, cls]
            for det in non_max_suppression_41c(out, conf_thres, iou_thres, labels=[], multi_label=True, agnostic=False)[0]:
                pt1, pt2, conf, cls = det[:2], det[2:4], det[4], int(det[5])
                color = canvas.color(cls)
                title = f"{str(cls) if not class_names else class_names[cls]}: {conf:.2f}"
                canvas.draw_box(pt1, pt2, alpha=0.4, thickness=-1, color=color)
                canvas.draw_box(pt1, pt2, color=color, title=title)
            canvas.show(canvas_title)

    def images(self, image_dir, conf_thres=0.25, iou_thres=0.45, inf_size=640):
        """
        [1~9]: jump to n% of dataset
        [A]: prev image
        [D]: next image
        [Q]: quit
        """
        # load image paths in queue
        test_queue = []
        for image in os.listdir(image_dir):
            if not image.startswith("._") and (image.endswith(".png") or image.endswith(".jpg")):
                fp = f"{image_dir}/{image}"
                test_queue.append(fp)
        # control loop
        i = 0
        canvas = Canvas()
        self.model.eval()
        while len(test_queue):
            if i >= len(test_queue):
                i = 0
            fp = test_queue[i]
            print(fp)
            image = cv2.imread(fp)
            image = uniform_scale(image, inf_size)
            self.detect(self.canvas_title, self.model, image, canvas, conf_thres, iou_thres, self.class_names)
            flag = cv2.waitKey(0)
            if flag == ord("q"):
                break
            elif flag == ord("a"):
                i -= 1
            elif flag in [ord(x) for x in "1234567890"]:
                i = int(len(test_queue) * (flag - ord("0")) / 10)
            else:
                i += 1

    def stream(self, data_stream: Iterable, conf_thres=0.25, iou_thres=0.45, auto=True, cache_size=10):
        """
        [A]: prev image
        [D]: next image
        [Q]: quit
        """
        assert cache_size >= 1

        canvas = Canvas()
        self.model.eval()
        if auto:
            for frame in data_stream:
                self.detect(self.canvas_title, self.model, frame, canvas, conf_thres, iou_thres, self.class_names)
        else:
            cache_queue = []
            cursor = 0
            for frame in data_stream:
                # Append image in cache
                if len(cache_queue) >= cache_size:
                    cursor -= 1
                    cache_queue.pop(0)
                cache_queue.append(deepcopy(frame))
                # Show image
                self.detect(self.canvas_title, self.model, frame, canvas, conf_thres, iou_thres, self.class_names)
                # Key control
                while True:
                    flag = cv2.waitKey(0)
                    if flag == ord("q"):  # exit
                        return
                    if flag == ord("a"):  # go back
                        if cursor > 0:
                            cursor -= 1
                            self.detect(self.canvas_title, self.model, cache_queue[cursor], canvas, conf_thres, iou_thres, self.class_names)
                    else:
                        cursor += 1
                        if cursor > len(cache_queue) - 1:  # next image
                            break
                        self.detect(self.canvas_title, self.model, cache_queue[cursor], canvas, conf_thres, iou_thres, self.class_names)
