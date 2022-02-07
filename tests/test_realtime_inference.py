import cv2
import torch
import torchvision.transforms as T
from multiprocessing import Queue, Process
import numpy as np
from nano.datasets.coco_box2d_visualize import draw_bounding_boxes, draw_center_points
from nano.datasets.class_utils import c26_classes
from nano.ops.box2d import non_max_suppression


def detection(conf_thres, iou_thres, inf_size, device, capture_queue, result_queue):
    model = acquire_model()
    model.eval().to(device)
    model.head.debug = True
    print("> model online.")
    transforms = T.Compose([T.ToPILImage(), T.Resize(inf_size), T.ToTensor()])
    with torch.no_grad():
        while True:
            if not capture_queue.empty():
                frame = capture_queue.get()
                # process image
                x = transforms(frame).to(device).unsqueeze(0)
                results, grid_mask, stride_mask = model(x)
                results = results[..., :4+3]
                centers = (grid_mask[0] + 0.5) * stride_mask[0].unsqueeze(-1)
                alphas = results[0, :, 4:].max(dim=-1).values
                mask = alphas >= iou_thres
                centers, alphas = centers[mask], alphas[mask]
                # Run NMS
                out = non_max_suppression(results, conf_thres, iou_thres, focal_nms=True)[0]  # batch 0
                result_queue.put((out, centers, alphas))


def test_video(capture_generator, capture_size, conf_thres, iou_thres, class_names, device="cpu", always_on_top=False):
    cap_h, cap_w = capture_size
    ratio = 448 / max(capture_size)  # h, w <= 416
    inf_h = int(np.ceil(cap_h * ratio / 64) * 64)  # (padding for Thinkpad-P51 front camera)
    inf_w = int(np.ceil(cap_w * ratio / 64) * 64)  # (padding for Thinkpad-P51 front camera)
    border_h = int((inf_h / ratio - cap_h) // 2)
    border_w = int((inf_w / ratio - cap_w) // 2)
    inference_size = (inf_h, inf_w)
    capture_queue = Queue(maxsize=1)
    result_queue = Queue(maxsize=64)
    bbox_set = []

    # inference process --------------
    proc_1 = Process(target=detection, args=[conf_thres, iou_thres, inference_size, device, capture_queue, result_queue])
    proc_1.daemon = True
    proc_1.start()

    # # capture process ----------------
    bbox_set = []
    for frame in capture_generator:
        frame = cv2.copyMakeBorder(frame, border_h, border_h, border_w, border_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        if capture_queue.empty():
            capture_queue.put(frame)
        if not result_queue.empty():  # update bbox_set
            bbox_set, centers, _ = result_queue.get()
        if len(bbox_set) == 0:
            print("nothing detected")
        else:
            x = bbox_set.clone()
            x[..., :4] /= ratio  # rescale to raw image size
            centers /= ratio  # rescale to raw image sizection
            box_classes = [class_names[n] for n in x[..., 5].cpu().int()]  # xyxy-conf-cls
            box_labels = [f"{cname} {conf:.2f}" for cname, conf in zip(box_classes, x[..., 4])]
            # draw bounding boxes with builtin opencv2 fu
            frame = draw_center_points(frame, centers, thickness=3)
            frame = draw_bounding_boxes(frame, boxes=x[..., :4], boxes_label=box_labels)
            # frame = draw_bounding_boxes(frame, boxes=x[..., :4], boxes_label=box_labels, alphas=x[..., 4])
        cv2.imshow("frame", frame)
        if always_on_top:
            cv2.setWindowProperty("frame", cv2.WND_PROP_TOPMOST, 1)
        if cv2.waitKey(10) == 27:
            return


def test_front_camera(conf_thres, iou_thres, class_names, device="cpu"):
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

        test_video(capture_fn(), capture_size, conf_thres, iou_thres, class_names, device)
    except Exception as e:
        capture.release()
        cv2.destroyAllWindows()
        raise e


def test_screenshot(conf_thres, iou_thres, class_names, device="cpu"):
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

        test_video(capture_fn(), capture_size, conf_thres, iou_thres, class_names, device, always_on_top=True)
    except Exception as e:
        cv2.destroyAllWindows()
        raise e


def test_yuv(conf_thres, iou_thres, class_names, device="cpu"):
    import os
    import time

    yuv_file = "../datasets/1280x720_4.yuv"
    yuv_h, yuv_w = 720, 1280

    try:
        # Number of frames: in YUV420 frame size in bytes is width*height*1.5
        file_size = os.path.getsize(yuv_file)
        n_frames = file_size // (yuv_w * yuv_h * 3 // 2)
        capture_size = (yuv_h, yuv_w)

        def capture_fn(fps=24):
            with open(yuv_file, "rb") as f:
                for _ in range(n_frames):
                    # Read Y, U and V color channels and reshape to height*1.5 x width numpy array
                    yuv = np.frombuffer(f.read(yuv_w * yuv_h * 3 // 2), dtype=np.uint8).reshape((yuv_h * 3 // 2, yuv_w))
                    # Convert YUV420 to BGR (for testing), applies BT.601 "Limited Range" conversion.
                    # frame = frame[:360, :640, :]
                    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                    yield frame
                    time.sleep(1 / fps)

        test_video(capture_fn(), capture_size, conf_thres, iou_thres, class_names, device)
    except Exception as e:
        cv2.destroyAllWindows()
        raise e


def acquire_model():
    from nano.models.model_zoo.nano_ghost import GhostNano_3x4_m96

    model = GhostNano_3x4_m96(num_classes=26)
    model.load_state_dict(torch.load("runs/train/exp10/last.pt", map_location="cpu")['state_dict'])
    return model


if __name__ == "__main__":
    test_front_camera(0.25, 0.45, c26_classes, device="cpu")
    # test_screenshot(0.25, 0.45, ["person", "bike", "car"], device="cpu")
    # test_yuv(0.3, 0.45, c26_classes, device="cpu")