import time
from unittest import result
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from multiprocessing import Queue, Process
import numpy as np
from nano.data.visualize import Canvas
from loguru import logger


def detection(model, conf_thres, iou_thres, inf_size, device, capture_queue, result_queue):
    model.eval().to(device)
    model.head.debug = True
    logger.info("Model Online")
    transforms = T.Compose([T.ToPILImage(), T.Resize(inf_size), T.ToTensor()])
    with torch.no_grad():
        while True:
            if not capture_queue.empty():
                frame = capture_queue.get()
                # process image
                x = transforms(frame).to(device).unsqueeze(0)
                results, grid_mask, stride_mask = model(x)
                # results[..., 4:] = results[..., 4:] * torch.exp(results[..., 4:]) / torch.exp(results[..., 4:]).sum(dim=-1, keepdim=True)
                centers = (grid_mask[0] + 0.5) * stride_mask[0].unsqueeze(-1)
                alphas = results[0, :, 4:].max(dim=-1).values
                mask = alphas >= conf_thres
                result_queue.put((centers[mask], stride_mask[0, mask].unsqueeze(-1).float(), alphas[mask]))


def test_video_activation(model, capture_generator, capture_size, conf_thres, iou_thres, class_names, device="cpu", always_on_top=False):
    cap_h, cap_w = capture_size
    ratio = 416 / max(capture_size)  # h, w <= 416
    inf_h = int(np.ceil(cap_h * ratio / 64) * 64)  # (padding for Thinkpad-P51 front camera)
    inf_w = int(np.ceil(cap_w * ratio / 64) * 64)  # (padding for Thinkpad-P51 front camera)
    border_h = int((inf_h / ratio - cap_h) // 2)
    border_w = int((inf_w / ratio - cap_w) // 2)
    inference_size = (inf_h, inf_w)
    capture_queue = Queue(maxsize=1)
    result_queue = Queue(maxsize=64)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(f"test_video_act_buf_{int(time.time())}.avi", fourcc, 24.0, (cap_w + 2 * border_w, cap_h + 2 * border_h))

    # inference process --------------
    proc_1 = Process(target=detection, args=[model, conf_thres, iou_thres, inference_size, device, capture_queue, result_queue])
    proc_1.daemon = True
    proc_1.start()
    result_buffer = None

    # # capture process ----------------
    for frame in capture_generator:
        frame = cv2.copyMakeBorder(frame, border_h, border_h, border_w, border_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        if capture_queue.empty():
            capture_queue.put(frame)
        if not result_queue.empty():  # update bbox_set
            result_buffer = result_queue.get()
        if result_buffer is not None:
            centers, stride_mask, alphas = result_buffer
            centers /= ratio
            stride_mask /= ratio
            logger.info(f"Rendering activation ({alphas.shape})")
            canvas = Canvas(frame)
            canvas.set_color((0, 0, 255))
            for (cx, cy), s, alpha in zip(centers, stride_mask, alphas):
                alpha = alpha * 2
                if alpha > 1:
                    alpha = 1
                canvas.draw_box((cx - 0.5 * s, cy - 0.5 * s, cx + 0.5 * s, cy + 0.5 * s), alpha=alpha, thickness=-1)
                frame = canvas.image
        cv2.imshow("frame", frame)
        out.write(frame)
        if always_on_top:
            cv2.setWindowProperty("frame", cv2.WND_PROP_TOPMOST, 1)
        if cv2.waitKey(10) == 27:
            break

    out.release()
    return


def test_front_camera_activation(model, conf_thres, iou_thres, class_names, device="cpu"):
    try:
        capture = cv2.VideoCapture(0)  # VideoCapture 读取本地视频和打开摄像头
        cap_h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 计算视频的高
        cap_w = capture.get(cv2.CAP_PROP_FRAME_WIDTH)  # 计算视频的宽
        capture_size = (cap_h, cap_w)

        def capture_fn():
            while True:
                ret, frame = capture.read()
                if ret is False:
                    break
                yield cv2.flip(frame, 1)  # cv2.flip 图像翻转

        test_video_activation(model, capture_fn(), capture_size, conf_thres, iou_thres, class_names, device)
    except Exception as e:
        capture.release()
        cv2.destroyAllWindows()
        raise e


def test_screenshot_activation(model, conf_thres, iou_thres, class_names, device="cpu"):
    from mss import mss
    import pyautogui as pag

    capture_range = {"top": 0, "left": 0, "width": 448, "height": 448}

    try:
        capture = mss()
        cap_h = capture_range["height"]
        cap_w = capture_range["width"]
        capture_size = (cap_h * 2, cap_w * 2)

        def capture_fn():
            while True:
                x, y = pag.position()  # 返回鼠标的坐标
                capture_range["top"] = y - capture_range["height"] // 2
                capture_range["left"] = x - capture_range["width"] // 2
                frame = capture.grab(capture_range)
                frame = np.array(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                yield frame

        test_video_activation(model, capture_fn(), capture_size, conf_thres, iou_thres, class_names, device, always_on_top=True)
    except Exception as e:
        cv2.destroyAllWindows()
        raise e


def test_yuv_activation(model, conf_thres, iou_thres, class_names, yuv_file, yuv_size=(720, 1280), fps=24, device="cpu"):
    import os
    import time

    yuv_h, yuv_w = yuv_size

    try:
        # Number of frames: in YUV420 frame size in bytes is width*height*1.5
        file_size = os.path.getsize(yuv_file)
        n_frames = file_size // (yuv_w * yuv_h * 3 // 2)
        capture_size = (yuv_h, yuv_w)

        def capture_fn(fps=fps):
            with open(yuv_file, "rb") as f:
                for _ in range(n_frames):
                    # Read Y, U and V color channels and reshape to height*1.5 x width numpy array
                    yuv = np.frombuffer(f.read(yuv_w * yuv_h * 3 // 2), dtype=np.uint8).reshape((yuv_h * 3 // 2, yuv_w))
                    # Convert YUV420 to BGR (for testing), applies BT.601 "Limited Range" conversion.
                    # frame = frame[:360, :640, :]
                    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                    yield frame
                    time.sleep(1 / fps)

        test_video_activation(model, capture_fn(), capture_size, conf_thres, iou_thres, class_names, device, always_on_top=True)
    except Exception as e:
        cv2.destroyAllWindows()
        raise e
