import cv2
import torch
import torchvision.transforms as T
from multiprocessing import Queue, Process
import numpy as np
from nano.datasets.coco_box2d_visualize import draw_bounding_boxes

from nano.ops.box2d import non_max_suppression


def cv2_draw_bbox(frame, x, class_names):
    # xyxy-conf-cls
    box_classes = [class_names[n] for n in x[..., 5].cpu().int()]
    box_labels = [f"{cname} {conf:.2f}" for cname, conf in zip(box_classes, x[..., 4])]
    return draw_bounding_boxes(
        image=frame,
        boxes=x[..., :4],
        boxes_label=box_labels,
        alphas=(x[..., 4] * 3).clamp(0, 1),
    )


def detection(conf_thres, iou_thres, inf_size, device, capture_queue, bbox_queue):
    model = acquire_model()
    model.eval().to(device)
    print("> model online.")
    transforms = T.Compose([T.ToPILImage(), T.Resize(inf_size), T.ToTensor()])
    while True:
        if not capture_queue.empty():
            frame = capture_queue.get()
            # process image
            x = transforms(frame)
            x = x.unsqueeze(0).to(device)
            with torch.no_grad():
                # Run model
                results = model(x)  # inference and training outputs
                # Run NMS
                out = non_max_suppression(results, conf_thres, iou_thres)[0]  # batch 0
            bbox_queue.put(out)


def test_with_capture_fn(capture_fn, capture_size, conf_thres, iou_thres, class_names, device="cpu"):
    cap_h, cap_w = capture_size
    ratio = 416 / max(capture_size)  # h, w <= 416
    inf_h = int(np.ceil(cap_h * ratio / 32) * 32)  # (padding for Thinkpad-P51 front camera)
    inf_w = int(np.ceil(cap_w * ratio / 32) * 32)  # (padding for Thinkpad-P51 front camera)
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
    for frame in capture_fn():
        frame = cv2.copyMakeBorder(frame, border_h, border_h, border_w, border_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        if capture_queue.empty():
            # print("put frame", capture_queue.qsize())
            capture_queue.put(frame)
        if not result_queue.empty():  # update bbox_set
            bbox_set = result_queue.get()
        if len(bbox_set) == 0:
            print("nothing detected")
        else:
            x = bbox_set.clone()
            x[..., :4] /= ratio  # rescale to raw image size
            frame = cv2_draw_bbox(frame, x, class_names)
        cv2.imshow("frame", frame)
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

        test_with_capture_fn(capture_fn, capture_size, conf_thres, iou_thres, class_names, device)
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
        capture_size = (cap_h, cap_w)

        def capture_fn():
            while True:
                x, y = pag.position() #返回鼠标的坐标
                capture_range['top'] = y - capture_range['height']//2
                capture_range['left'] = x - capture_range['width']//2
                frame = capture.grab(capture_range)
                frame = np.array(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                yield frame

        test_with_capture_fn(capture_fn, capture_size, conf_thres, iou_thres, class_names, device)
    except Exception as e:
        cv2.destroyAllWindows()
        raise e




def test_yuv(conf_thres, iou_thres, class_names, device="cpu"):
    import os
    import time

    yuv_file = "1280x720_3.yuv"
    yuv_h, yuv_w = 720, 1280

    try:
        # Number of frames: in YUV420 frame size in bytes is width*height*1.5
        file_size = os.path.getsize(yuv_file)
        n_frames = file_size // (yuv_w * yuv_h * 3 // 2)
        capture_size = (yuv_h, yuv_w)

        def capture_fn(fps=12):
            with open(yuv_file, "rb") as f:
                for _ in range(n_frames):
                    # Read Y, U and V color channels and reshape to height*1.5 x width numpy array
                    yuv = np.frombuffer(f.read(yuv_w * yuv_h * 3 // 2), dtype=np.uint8).reshape((yuv_h * 3 // 2, yuv_w))
                    # Convert YUV420 to BGR (for testing), applies BT.601 "Limited Range" conversion.
                    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                    yield frame
                    time.sleep(1 / fps)

        test_with_capture_fn(capture_fn, capture_size, conf_thres, iou_thres, class_names, device)
    except Exception as e:
        cv2.destroyAllWindows()
        raise e


#                 frame = frame[:360, :640, :]



def acquire_model():
    from nano.models.model_zoo.nano_ghost import GhostNano_3x3_m96

    model = GhostNano_3x3_m96(num_classes=3)
    model.load_state_dict(torch.load("release/GhostNano_3x3_m96/GhostNano_3x3_m96.pt", map_location="cpu"))
    return model


if __name__ == "__main__":
    test_yuv(
        0.2,
        0.45,
        ["person", "bike", "car"],
        device="cpu",
    )
