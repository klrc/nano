import sys
import cv2
import torch
import torchvision.transforms as T
from multiprocessing import Queue, Process
import numpy as np
from loguru import logger
import time

sys.path.append(".")
from nano.data.visualize import Canvas  # noqa: E402
from nano.models.box2d import non_max_suppression  # noqa: E402


def detection(model, conf_thres, iou_thres, inf_size, device, capture_queue, result_queue):
    model = model.eval().to(device)
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
                centers = (grid_mask[0] + 0.5) * stride_mask[0].unsqueeze(-1)
                alphas = results[0, :, 4:].max(dim=-1).values
                mask = alphas >= conf_thres
                centers, alphas = centers[mask], alphas[mask]
                # Run NMS
                out = non_max_suppression(results, conf_thres, iou_thres)[0]  # batch 0
                result_queue.put((out, centers, alphas))


def test_video(model, capture_generator, capture_size, conf_thres, iou_thres, class_names, device="cpu", always_on_top=False):
    cap_h, cap_w = capture_size
    ratio = 512 / max(capture_size)  # h, w <= 416
    inf_h = int(np.ceil(cap_h * ratio / 64) * 64)  # (padding for Thinkpad-P51 front camera)
    inf_w = int(np.ceil(cap_w * ratio / 64) * 64)  # (padding for Thinkpad-P51 front camera)
    border_h = int((inf_h / ratio - cap_h) // 2)
    border_w = int((inf_w / ratio - cap_w) // 2)
    inference_size = (inf_h, inf_w)
    capture_queue = Queue(maxsize=1)
    result_queue = Queue(maxsize=64)
    bbox_set = []

    record_mode = False

    if record_mode:
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        video_out = cv2.VideoWriter(f"test_video_act_buf_{int(time.time())}.mp4", fourcc, 24.0, (cap_w + 2 * border_w, cap_h + 2 * border_h))

    # inference process --------------
    proc_1 = Process(target=detection, args=[model, conf_thres, iou_thres, inference_size, device, capture_queue, result_queue])
    proc_1.daemon = True
    proc_1.start()

    # # capture process ----------------
    bbox_set = []
    for frame in capture_generator:
        frame = cv2.copyMakeBorder(frame, border_h, border_h, border_w, border_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        if capture_queue.empty():
            capture_queue.put(frame)
        if not result_queue.empty():  # update bbox_set
            bbox_set, centers, alphas = result_queue.get()
        if len(bbox_set) > 0:
            logger.info(f"Receving frame with {len(bbox_set)} objects")
            x = bbox_set.clone()
            x[..., :4] /= ratio  # rescale to raw image size
            centers /= ratio  # rescale to raw image sizection
            box_classes = [class_names[n] for n in x[..., 5].cpu().int()]  # xyxy-conf-cls
            box_labels = [f"{cname} {conf:.2f}" for cname, conf in zip(box_classes, x[..., 4])]
            # draw bounding boxes with builtin opencv2
            canvas = Canvas(frame)
            for box, label in zip(x[..., :4], box_labels):
                canvas.draw_box(box)
                canvas.draw_text_with_background(label, (box[0], box[1]))
            for center, alpha in zip(centers, alphas):
                canvas.draw_point(center, alpha=alpha)
            frame = canvas.image

        if record_mode:
            video_out.write(frame)
        else:
            cv2.imshow("frame", frame)
            if always_on_top:
                cv2.setWindowProperty("frame", cv2.WND_PROP_TOPMOST, 1)
            if cv2.waitKey(10) == 27:
                return

    if record_mode:
        video_out.release()


def test_front_camera(model, conf_thres, iou_thres, class_names, device="cpu"):
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

        test_video(model, capture_fn(), capture_size, conf_thres, iou_thres, class_names, device)
    except Exception as e:
        capture.release()
        cv2.destroyAllWindows()
        raise e


def test_screenshot(model, conf_thres, iou_thres, class_names, device="cpu"):
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

        test_video(model, capture_fn(), capture_size, conf_thres, iou_thres, class_names, device, always_on_top=True)
    except Exception as e:
        cv2.destroyAllWindows()
        raise e


def test_yuv(model, conf_thres, iou_thres, class_names, yuv_file, yuv_size=(720, 1280), fps=24, device="cpu"):
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

        test_video(model, capture_fn(), capture_size, conf_thres, iou_thres, class_names, device, always_on_top=True)
    except Exception as e:
        cv2.destroyAllWindows()
        raise e


if __name__ == "__main__":
    from nano.data.dataset_info import drive3_names
    from nano.models.model_zoo import GhostNano_3x3_m96, GhostNano_3x3_l128

    model = GhostNano_3x3_m96(len(drive3_names))
    model.load_state_dict(torch.load("runs/train/exp29/best.pt", map_location="cpu")['state_dict'])
    test_yuv(model, 0.25, 0.45, drive3_names, yuv_file="../datasets/1280x720_4.yuv", device="cpu")
