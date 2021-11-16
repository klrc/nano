import cv2
import torch
from torch._C import device
import torchvision.transforms as T
import nano
from multiprocessing import Queue, Process


def detection_iou(box1, box2):
    area_a = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_b = (box2[2] - box2[0]) * (box2[3] - box2[1])
    w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    h = min(box1[3], box2[3]) - max(box1[1], box2[1])
    if w <= 0 or h <= 0:
        return 0
    area_c = w * h
    return area_c / (area_a + area_b - area_c)


def detection_nms(preds, conf_thres, iou_thres, max_nms=30000):
    """Runs Non-Maximum Suppression (NMS) on inference results
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        conf_thres = 0.2  # confidence threshold
        iou_thres = 0.45  # iou threshold
    Returns:
         list of detections, on (n,8) tensor per image [xyxy, conf, cls, cls_conf, obj_conf]
    """

    # Settings
    results = []

    # Reformat data
    for pred in preds:
        obj_conf = pred[4]
        cls_conf, cls_id = pred[5:].max(dim=0, keepdim=True)  # Confidence calculation
        conf = obj_conf * cls_conf
        if conf < conf_thres:  # Select candidates
            continue
        x1 = pred[0] - pred[2] / 2.0  # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        y1 = pred[1] - pred[3] / 2.0
        x2 = pred[0] + pred[2] / 2.0
        y2 = pred[1] + pred[3] / 2.0
        results.append([x1, y1, x2, y2, conf, cls_id, cls_conf, obj_conf])  # Detections matrix nx8

    # Check results
    n = len(results)  # number of boxes
    if n == 0:  # no boxes
        return results
    results = sorted(results, key=lambda x: x[4], reverse=True)  # sort by confidence
    if n > max_nms:  # excess boxes
        results = results[:max_nms]
        n = max_nms

    # Batched NMS
    targets = [i for i in range(n)]
    keeped_targets = []
    while len(targets) > 0:
        t = targets.pop(0)
        keeped_targets.append(t)
        for i, x in enumerate(targets):
            if detection_iou(results[x], results[t]) > iou_thres:
                targets[i] = -1  # abandon
        targets = [i for i in targets if i != -1]
    return [results[i] for i in keeped_targets]


def detection_post_process(preds, anchors, strides, num_classes, conf_thres, iou_thres):
    # reshape outputs
    results = []
    anchors = torch.tensor(anchors).view(len(strides), 1, 1, 1, 2)
    for i, pred in enumerate(preds):  # predictions with different scales
        y = pred.sigmoid()
        y = y.permute(0, 2, 3, 1)
        bs, _, ny, nx = pred.shape
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        grid = torch.stack((xv, yv), 2).view((1, ny, nx, 2)).float()
        y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + grid) * strides[i]  # x, y
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchors[i]  # w, h
        results.append(y.view(bs, -1, num_classes + 5))
    results = torch.cat(results, 1)
    results = results.squeeze(0)
    results = detection_nms(results, conf_thres, iou_thres)
    return results


def detection(conf_thres, iou_thres, device, capture_queue, bbox_queue):
    model = acquire_model()
    model.eval().to(device)
    print("> model online.")
    anchors = list(model.head.anchor_grid.flatten().numpy())
    strides = list(model.head.stride.numpy())
    num_classes = model.head.nc
    transforms = T.Compose([T.ToPILImage(), T.Resize((224, 416)), T.ToTensor()])
    while True:
        if not capture_queue.empty():
            frame = capture_queue.get()
            # process image
            x = transforms(frame)
            results = model(x.unsqueeze(0).to(device))
            results = [x.cpu() for x in results]
            results = detection_post_process(results, anchors, strides, num_classes, conf_thres, iou_thres)
            results = [[float(xi) for xi in x] for x in results]
            bbox_queue.put(results)
            # print(f'\r> {len(results)} objects detected.', end='')


def cv2_draw_bbox(frame, x, canvas_h, canvas_w, class_names):
    x1, y1, x2, y2, conf, cls_id, cls_conf, obj_conf = x
    x1 = int(x1 / 416 * canvas_w)
    x2 = int(x2 / 416 * canvas_w)
    y1 = int(y1 / 224 * canvas_h)
    y2 = int(y2 / 224 * canvas_h)
    label = f"detected: {class_names[int(cls_id)]} conf={conf:.2%} cls={cls_conf:.2%} obj={obj_conf:.2%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    cv2.rectangle(frame, (x1, y1), (x2, y2), (75, 0, 130), 1, cv2.LINE_AA, 0)
    text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(frame, (x1, y1), (x1 + text_w + 2, y1 + text_h + 2), (75, 0, 130), -1)
    cv2.putText(frame, label, (x1, y1 + text_h), font, font_scale, (255, 255, 255), font_thickness)


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
        while True:
            ret, frame = capture.read()
            frame = cv2.copyMakeBorder(frame, 0, 0, 125, 125, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            if ret is False:
                break
            frame = cv2.flip(frame, 1)  # cv2.flip 图像翻转
            if capture_queue.empty():
                # print("put frame", capture_queue.qsize())
                capture_queue.put(frame)
            if not result_queue.empty():
                bbox_set = result_queue.get()
            if len(bbox_set) > 0:
                for x in bbox_set:
                    cv2_draw_bbox(frame, x, canvas_h, canvas_w, class_names)
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
            if len(bbox_set) > 0:
                for x in bbox_set:
                    cv2_draw_bbox(frame, x, canvas_h, canvas_w, class_names)
            cv2.imshow("frame", frame)
            if cv2.waitKey(10) == 27:
                break
    except Exception as e:
        cv2.destroyAllWindows()
        raise e


if __name__ == "__main__":

    def acquire_model():
        model = nano.models.yolox_esmk_shrink_misc(num_classes=4)
        model.load_state_dict(torch.load("runs/train/exp139/weights/last.pt", map_location="cpu")["state_dict"])
        model.dsp()
        return model

    test_front_camera(
        conf_thres=0.2,
        iou_thres=0.45,
        class_names=["person", "bike", "car", "misc"],
        device="cpu",
    )
