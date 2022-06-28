import os
import cv2

from ...io.canvas import Canvas
from ...io.utils import im2tensor


def read_labels_as_xyxy(lb):
    if not os.path.exists(lb):
        return []
    # read labels as c, x1, y1, x2, y2
    with open(lb, "r") as f:
        labels = []
        for line in f.readlines():
            c, x, y, w, h = [float(data) for data in line.split(" ")]
            labels.append([int(c), x - w / 2, y - h / 2, x + w / 2, y + h / 2])
        return labels


def travel_dataset(path, class_names=None):
    """
    1~9: jump to n% of dataset
    a: prev image
    *: next image
    q: quit
    """
    # load image paths in queue
    test_queue = []
    for image in os.listdir(path):
        if not image.startswith("._") and (image.endswith(".png") or image.endswith(".jpg")):
            fp = f"{path}/{image}"
            test_queue.append(fp)
    # control loop
    i = 0
    canvas = Canvas()
    while i < len(test_queue):
        fp = test_queue[i]
        print(fp)
        image = cv2.imread(fp)
        lb = fp.replace(".png", ".txt").replace(".jpg", ".txt").replace("/images", "/labels")
        lb = read_labels_as_xyxy(lb)
        image = im2tensor(image).unsqueeze(0)
        height, width = image.shape[-2:]
        canvas.load(image[0])
        for cls, x1, y1, x2, y2 in lb:
            pt1, pt2 = (x1 * width, y1 * height), (x2 * width, y2 * height)
            color = canvas.color(cls)
            title = str(cls) if not class_names else class_names[cls]
            canvas.draw_box(pt1, pt2, alpha=0.4, thickness=-1, color=color)
            canvas.draw_box(pt1, pt2, color=color, title=title)
        canvas.show("ground truth")
        flag = cv2.waitKey(0)
        if flag == ord("q"):
            break
        elif flag == ord("a"):
            i -= 1
        elif flag in [ord(x) for x in "1234567890"]:
            i = int(len(test_queue) * (flag - ord("0")) / 10)
        else:
            i += 1
