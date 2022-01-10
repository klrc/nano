import cv2
import torch
import torchvision.transforms as T
from multiprocessing import Queue, Process
import numpy as np
from nano.datasets.coco_box2d_visualize import draw_bounding_boxes

from nano.ops.box2d import non_max_suppression


TARGET_SIZE_WIDTH = 416
TARGET_SIZE_HEIGHT = 224


def cv2_draw_bbox(frame, x, canvas_h, canvas_w, class_names):
    # xyxy-conf-cls
    x[..., 0] *= canvas_w / TARGET_SIZE_WIDTH
    x[..., 1] *= canvas_h / TARGET_SIZE_HEIGHT
    x[..., 2] *= canvas_w / TARGET_SIZE_WIDTH
    x[..., 3] *= canvas_h / TARGET_SIZE_HEIGHT
    box_classes = [class_names[n] for n in x[..., 5].cpu().int()]
    box_labels = [f'{cname} {conf:.2f}' for cname, conf in zip(box_classes, x[...,4])]
    return draw_bounding_boxes(
        image=frame,
        boxes=x[..., :4],
        boxes_label=box_labels,
    )


def detection(conf_thres, iou_thres, device, capture_queue, bbox_queue):
    model = acquire_model()
    model.eval().to(device)
    print("> model online.")
    transforms = T.Compose([T.ToPILImage(), T.Resize((TARGET_SIZE_HEIGHT, TARGET_SIZE_WIDTH)), T.ToTensor()])
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


def test_front_camera(conf_thres, iou_thres, class_names, device="cpu"):
    capture = cv2.VideoCapture(0)  # VideoCapture 读取本地视频和打开摄像头
    canvas_h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 计算视频的高
    canvas_w = capture.get(cv2.CAP_PROP_FRAME_WIDTH) + 125 * 2  # 计算视频的宽 (padding for Thinkpad-P51 front camera)
    capture_queue = Queue(maxsize=1)
    result_queue = Queue(maxsize=64)
    bbox_set = []

    try:
        # inference process --------------
        proc_1 = Process(target=detection, args=[conf_thres, iou_thres, device, capture_queue, result_queue])
        proc_1.daemon = True
        proc_1.start()

        # # capture process ----------------
        bbox_set = []
        while True:
            ret, frame = capture.read()
            frame = cv2.copyMakeBorder(frame, 0, 0, 125, 125, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            if ret is False:
                break
            frame = cv2.flip(frame, 1)  # cv2.flip 图像翻转
            if capture_queue.empty():
                # print("put frame", capture_queue.qsize())
                capture_queue.put(frame)
            if not result_queue.empty():  # update bbox_set
                bbox_set = result_queue.get()
            if len(bbox_set) == 0:
                print("nothing detected")
            else:
                cv2_draw_bbox(frame, bbox_set.clone(), canvas_h, canvas_w, class_names)
            cv2.imshow("frame", frame)
            if cv2.waitKey(10) == 27:
                break
    except Exception as e:
        capture.release()
        cv2.destroyAllWindows()
        raise e


def test_screenshot(conf_thres, iou_thres, class_names, device="cpu"):
    from mss import mss
    import numpy as np

    canvas_x = 0
    canvas_y = 0
    canvas_h = 840
    canvas_w = 1560
    bounding_box = {"top": canvas_y, "left": canvas_x, "width": canvas_w, "height": canvas_h}
    capture_queue = Queue(maxsize=1)
    result_queue = Queue(maxsize=64)
    capture = mss()
    bbox_set = []

    try:
        # inference process --------------
        proc_1 = Process(target=detection, args=[conf_thres, iou_thres, device, capture_queue, result_queue])
        proc_1.daemon = True
        proc_1.start()

        # # capture process ----------------
        while True:
            frame = capture.grab(bounding_box)
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            if capture_queue.empty():
                # print("put frame", capture_queue.qsize())
                capture_queue.put(frame)
            if not result_queue.empty():
                bbox_set = result_queue.get()
                cv2_draw_bbox(frame, bbox_set, canvas_h, canvas_w, class_names)
            cv2.imshow("frame", frame)
            if cv2.waitKey(10) == 27:
                break
    except Exception as e:
        cv2.destroyAllWindows()
        raise e


def test_yuv(file_name, height, width, conf_thres, iou_thres, class_names, device="cpu", fps=12):
    import os
    import time

    capture_queue = Queue(maxsize=1)
    result_queue = Queue(maxsize=64)
    bbox_set = []

    try:
        # inference process --------------
        proc_1 = Process(target=detection, args=[conf_thres, iou_thres, device, capture_queue, result_queue])
        proc_1.daemon = True
        proc_1.start()

        # # capture process ----------------

        # Number of frames: in YUV420 frame size in bytes is width*height*1.5
        file_size = os.path.getsize(file_name)
        n_frames = file_size // (width * height * 3 // 2)
        time.sleep(3)

        with open(file_name, "rb") as f:
            for _ in range(n_frames):
                # Read Y, U and V color channels and reshape to height*1.5 x width numpy array
                yuv = np.frombuffer(f.read(width * height * 3 // 2), dtype=np.uint8).reshape((height * 3 // 2, width))
                # Convert YUV420 to BGR (for testing), applies BT.601 "Limited Range" conversion.
                frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                frame = frame[:360, :640, :]
                if capture_queue.empty():
                    # print("put frame", capture_queue.qsize())
                    capture_queue.put(frame)
                if not result_queue.empty():
                    bbox_set = result_queue.get()
                    cv2_draw_bbox(frame, bbox_set, height // 2, width // 2, class_names)
                cv2.imshow("frame", frame)
                time.sleep(1 / fps)
                if cv2.waitKey(10) == 27:
                    break
    except Exception as e:
        cv2.destroyAllWindows()
        raise e


if __name__ == "__main__":

    def acquire_model():
        from nano.models.model_zoo.nano_ghost import GhostNano_3x3_s32

        model = GhostNano_3x3_s32(num_classes=3)
        model.load_state_dict(torch.load("runs/train/exp119/last.pt", map_location="cpu")["state_dict"])
        return model

    test_front_camera(
        0.4,
        0.45,
        ["person", "bike", "car"],
        device="cpu",
    )
    # test_yuv(
    #     "1280x720_3.yuv",
    #     720,
    #     1280,
    #     conf_thres=0.2,
    #     iou_thres=0.6,
    #     class_names=["person", "bike", "car"],
    #     device="cuda:0",
    #     fps=12,
    # )
