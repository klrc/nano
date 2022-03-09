import sys
import cv2
import torch
import torchvision.transforms as T
from multiprocessing import Queue, Process
import numpy as np
from loguru import logger
import time
import os
from mss import mss
import pyautogui as pag

sys.path.append(".")
from nano.data.visualize import Canvas  # noqa: E402
from nano.models.multiplex.box2d import non_max_suppression  # noqa: E402


default_classes = "person|bike|motorcycle|car|bus|truck|OOD".split("|")


def single_frame_inference(frame_queue, transforms, model, output_queue, device, conf_threshold, iou_threshold):
    # setup model
    model = model.eval().to(device)
    logger.info("Model Online")
    # start loop
    with torch.no_grad():
        while True:
            if not frame_queue.empty():  # fetch frame
                frame, r = frame_queue.get()
                frame = transforms(frame).to(device).unsqueeze(0)
                prediction = model(frame)
                prediction = non_max_suppression(prediction, conf_threshold, iou_threshold)
                prediction = prediction[0]  # single-frame only
                prediction[..., :4] /= r  # rescale to raw image size
                output_queue.put(prediction)


class VideoDetector:
    """
    A real-time video object detector for testing
    """

    def __init__(
        self,
        model,
        inference_size=416,
        max_stride=64,
        device="cpu",
    ):
        self.model = model
        self.inference_size = inference_size
        self.max_stride = max_stride
        self.device = device

    def deploy(self, conf_threshold=0.25, iou_threshold=0.45):
        # setup multiprocessing queues
        self.frame_queue = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=64)
        transforms = T.Compose([T.ToPILImage(), T.ToTensor()])
        # start inference process
        inference_args = [self.frame_queue, transforms, self.model, self.output_queue]
        inference_args += [self.device, conf_threshold, iou_threshold]
        proc_1 = Process(target=single_frame_inference, args=inference_args)
        proc_1.daemon = True
        proc_1.start()

    def feed(self, frame):
        # letterbox reshape function for inference
        h, w, _ = frame.shape
        r = self.inference_size / max(h, w)  # h, w <= 416
        inf_h = int(np.ceil(h * r / self.max_stride) * self.max_stride)  # (padding for Thinkpad-P51 front camera)
        inf_w = int(np.ceil(w * r / self.max_stride) * self.max_stride)  # (padding for Thinkpad-P51 front camera)
        border_h = int((inf_h / r - h) // 2)
        border_w = int((inf_w / r - w) // 2)
        frame = cv2.copyMakeBorder(frame, border_h, border_h, border_w, border_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        frame = cv2.resize(frame, (inf_w, inf_h), interpolation=cv2.INTER_AREA)
        # feed frame to frame_queue
        if self.frame_queue.empty():
            self.frame_queue.put((frame, r))  # skip current frame if model is busy

    def fetch(self):
        # fetch output from output_queue
        if not self.output_queue.empty():
            prediction = self.output_queue.get()
            if len(prediction) > 0:
                # update prediction
                logger.info(f"Receving frame with {len(prediction)} objects")
                return prediction
        return None


class VideoRenderPipe:
    def __init__(self, detector: VideoDetector, class_names, always_on_top=True) -> None:
        self.detector = detector
        self.always_on_top = always_on_top
        self.class_names = class_names
        self.detector.deploy()

    def feed(self, frame):
        # check if any output to render
        prediction = self.detector.fetch()
        # feed frames & save to buffer for rendering
        self.detector.feed(frame)
        # bounding box render pipeline
        if prediction is not None:
            box_classes = [self.class_names[n] for n in prediction[..., 5].cpu().int()]  # xyxy-conf-cls
            box_labels = [f"{cname} {conf:.2f}" for cname, conf in zip(box_classes, prediction[..., 4])]
            canvas = Canvas(frame)  # draw bounding boxes with builtin opencv2
            for box, label in zip(prediction[..., :4], box_labels):
                canvas.next_color()
                canvas.draw_box(box)
                canvas.draw_text_with_background(label, (box[0], box[1]))
            frame = canvas.image
        cv2.imshow("frame", frame)
        if self.always_on_top:
            cv2.setWindowProperty("frame", cv2.WND_PROP_TOPMOST, 1)


class FrontCameraDetection:
    def __init__(self, model, inference_size=416, max_stride=64, class_names=default_classes, always_on_top=True, device="cpu") -> None:
        detector = VideoDetector(model, inference_size, max_stride, device)
        self.pipe = VideoRenderPipe(detector, class_names, always_on_top)

    def start(self):
        try:
            capture = cv2.VideoCapture(0)  # VideoCapture 读取本地视频和打开摄像头
            # fourcc = cv2.VideoWriter_fourcc(*"MP4V")
            # video_out = cv2.VideoWriter(f"test_video_act_buf_{int(time.time())}.mp4", fourcc, 24.0, (cap_w + 2 * border_w, cap_h + 2 * border_h))
            while True:
                _, frame = capture.read()
                frame = cv2.flip(frame, 1)  # cv2.flip 图像翻转
                self.pipe.feed(frame)
                # start realtime render
                if cv2.waitKey(10) == 27:
                    break
        except Exception as e:
            cv2.destroyAllWindows()
            raise e
        finally:
            capture.release()


class ScreenshotDetection:
    def __init__(self, model, inference_size=416, max_stride=64, class_names=default_classes, always_on_top=True, device="cpu") -> None:
        detector = VideoDetector(model, inference_size, max_stride, device)
        self.pipe = VideoRenderPipe(detector, class_names, always_on_top)

    def start(self, size=(448, 448), flip=True):
        try:
            capture_range = {"top": 0, "left": 0, "width": size[0], "height": size[1]}
            capture = cv2.VideoCapture(0)  # VideoCapture 读取本地视频和打开摄像头
            capture = mss()
            while True:
                x, y = pag.position()  # 返回鼠标的坐标
                capture_range["top"] = y - capture_range["height"] // 2
                capture_range["left"] = x - capture_range["width"] // 2
                frame = capture.grab(capture_range)
                frame = np.array(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                if flip:
                    frame = cv2.flip(frame, 1)  # cv2.flip 图像翻转
                self.pipe.feed(frame)
                # start realtime render
                if cv2.waitKey(10) == 27:
                    break
        except Exception as e:
            cv2.destroyAllWindows()
            raise e


class YUVDetection:
    def __init__(self, model, inference_size=416, max_stride=64, class_names=default_classes, always_on_top=True, device="cpu") -> None:
        detector = VideoDetector(model, inference_size, max_stride, device)
        self.pipe = VideoRenderPipe(detector, class_names, always_on_top)

    def start(self, yuv_file, yuv_size=(720, 1280), fps=24):
        try:

            yuv_h, yuv_w = yuv_size
            # Number of frames: in YUV420 frame size in bytes is width*height*1.5
            file_size = os.path.getsize(yuv_file)
            n_frames = file_size // (yuv_w * yuv_h * 3 // 2)
            with open(yuv_file, "rb") as f:
                for _ in range(n_frames):
                    # Read Y, U and V color channels and reshape to height*1.5 x width numpy array
                    yuv = np.frombuffer(f.read(yuv_w * yuv_h * 3 // 2), dtype=np.uint8).reshape((yuv_h * 3 // 2, yuv_w))
                    # Convert YUV420 to BGR (for testing), applies BT.601 "Limited Range" conversion.
                    # frame = frame[:360, :640, :]
                    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                    self.pipe.feed(frame)
                    time.sleep(1 / fps)
                    # start realtime render
                    if cv2.waitKey(10) == 27:
                        break
        except Exception as e:
            cv2.destroyAllWindows()
            raise e


if __name__ == "__main__":
    from nano.models.model_zoo import GhostNano_3x3_m96

    model = GhostNano_3x3_m96(4)
    # model.load_state_dict(torch.load("runs/train/exp29/best.pt", map_location="cpu")['state_dict'])
    # detector = FrontCameraDetection(model, device="cpu")
    # detector.start()
    # detector = ScreenshotDetection(model, device="cpu")
    # detector.start(flip=False)
    detector = YUVDetection(model, device="cpu")
    detector.start(yuv_file="../datasets/1280x720_3.yuv")
