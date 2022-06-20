from multiprocessing import Queue, Process
import os
import time

import cv2
import numpy as np


class VideoLoader:
    def __init__(self) -> None:
        pass

    @staticmethod
    def collect(pipe: Queue, size, file_path, fps, color_format, non_block):
        raise NotImplementedError

    def play(self, pipe: Queue, fps=24, color_format=None, non_block=True):
        # Create a virtual video stream
        proc = Process(target=self.collect, args=(pipe, self.size, self.file_path, fps, color_format, non_block))
        proc.daemon = True
        proc.start()


class YUV420_VID_LOADER(VideoLoader):
    def __init__(self, file_path, size=(720, 1280)) -> None:
        super().__init__()
        self.file_path = file_path
        self.size = size

    @staticmethod
    def collect(pipe: Queue, size, file_path, fps, color_format, non_block):
        yuv_h, yuv_w = size
        # Number of frames: in YUV420 frame size in bytes is width*height*1.5
        file_size = os.path.getsize(file_path)
        n_frames = file_size // (yuv_w * yuv_h * 3 // 2)
        with open(file_path, "rb") as f:
            for _ in range(n_frames):
                # Read Y, U and V color channels and reshape to height*1.5 x width numpy array
                yuv = np.frombuffer(f.read(yuv_w * yuv_h * 3 // 2), dtype=np.uint8).reshape((yuv_h * 3 // 2, yuv_w))
                # Convert YUV420 to BGR (for testing), applies BT.601 "Limited Range" conversion.
                if not (non_block and pipe.full()):
                    pipe.put(yuv if color_format is None else cv2.cvtColor(yuv, color_format))
                time.sleep(1 / fps)
        # Ending flag as None
        pipe.put(None)

    def play(self, pipe: Queue, fps=24, color_format=cv2.COLOR_YUV2BGR_I420, non_block=True):
        # Create a virtual video stream
        proc = Process(target=self.collect, args=(pipe, self.size, self.file_path, fps, color_format, non_block))
        proc.daemon = True
        proc.start()


class H264_LOADER(VideoLoader):
    def __init__(self, file_path, size=None) -> None:
        super().__init__()
        self.file_path = file_path
        self.size = size

    @staticmethod
    def collect(pipe: Queue, size, file_path, fps, color_format, non_block):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            os._exit(-1)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if size:
                frame = cv2.resize(frame, size)
            if not (non_block and pipe.full()):
                pipe.put(frame if color_format is None else cv2.cvtColor(frame, color_format))
            time.sleep(1 / fps)
        # Ending flag as None
        pipe.put(None)


if __name__ == "__main__":
    # Test scripts
    loader = H264_LOADER("../datasets/6630-V1.5.7.0误报&漏报视频2000613/误报/hand.h264", (480, 384))
    pipe = Queue(maxsize=2)
    loader.play(pipe, fps=24)

    frame = pipe.get()
    while frame is not None:
        cv2.imshow("frame", frame)
        cv2.setWindowProperty("frame", cv2.WND_PROP_TOPMOST, 1)
        frame = pipe.get()
        # Press ESC to break
        if cv2.waitKey(10) == 27:
            break
