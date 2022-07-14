import os
import cv2
import numpy as np


class YUV400VideoStream:
    def __init__(self, file_path, size=(720, 1280), dsize=None, color_format=cv2.COLOR_GRAY2BGR) -> None:
        super().__init__()
        self.file_path = file_path
        self.size = size
        self.dsize = dsize
        self.color_format = color_format
        # Number of frames: in YUV420 frame size in bytes is width*height*1.5
        yuv_h, yuv_w = size
        file_size = os.path.getsize(file_path)
        self.max_frame = file_size // (yuv_w * yuv_h) - 1
        self.cur_frame = 0
        self.probe = None

    def __next__(self):
        self.cur_frame += 1
        if self.cur_frame > self.max_frame:
            self.probe.close()
            raise StopIteration
        yuv_h, yuv_w = self.size
        # Read Y, U and V color channels and reshape to height*1.5 x width numpy array
        yuv = np.frombuffer(self.probe.read(yuv_w * yuv_h), dtype=np.uint8).reshape((yuv_h, yuv_w))
        # Convert YUV400 to BGR (for testing), applies BT.601 "Limited Range" conversion.
        if self.color_format:
            yuv = cv2.cvtColor(yuv, self.color_format)
        if self.dsize:
            yuv = cv2.resize(yuv, self.dsize)
        return yuv

    def __iter__(self):
        self.cur_frame = 0
        self.probe = open(self.file_path, "rb")
        return self


class YUV420VideoStream:
    def __init__(self, file_path, size=(720, 1280), dsize=None, color_format=cv2.COLOR_YUV2BGR_I420) -> None:
        super().__init__()
        self.file_path = file_path
        self.size = size
        self.dsize = dsize
        self.color_format = color_format
        # Number of frames: in YUV420 frame size in bytes is width*height*1.5
        yuv_h, yuv_w = size
        file_size = os.path.getsize(file_path)
        self.max_frame = file_size // (yuv_w * yuv_h * 3 // 2) - 1
        self.cur_frame = 0
        self.probe = None

    def __next__(self):
        self.cur_frame += 1
        if self.cur_frame > self.max_frame:
            self.probe.close()
            raise StopIteration
        yuv_h, yuv_w = self.size
        # Read Y, U and V color channels and reshape to height*1.5 x width numpy array
        yuv = np.frombuffer(self.probe.read(yuv_w * yuv_h * 3 // 2), dtype=np.uint8).reshape((yuv_h * 3 // 2, yuv_w))
        # Convert YUV400 to BGR (for testing), applies BT.601 "Limited Range" conversion.
        if self.color_format:
            yuv = cv2.cvtColor(yuv, self.color_format)
        if self.dsize:
            yuv = cv2.resize(yuv, self.dsize)
        return yuv

    def __iter__(self):
        self.cur_frame = 0
        self.probe = open(self.file_path, "rb")
        return self
