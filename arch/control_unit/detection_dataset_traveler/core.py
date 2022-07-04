import os
import cv2
from xml.dom.minidom import parse


from ...io.canvas import Canvas
from ...io.converter import im2tensor


def read_labels_as_xyxy(lb, image_shape, class_names):
    if not os.path.exists(lb):
        return []
    # read labels as c, x1, y1, x2, y2
    labels = []
    if lb.endswith(".txt"):  # txt file
        with open(lb, "r") as f:
            for line in f.readlines():
                c, x, y, w, h = [float(data) for data in line.split(" ")]
                title = c if not class_names else class_names[int(c)]
                labels.append([title, x - w / 2, y - h / 2, x + w / 2, y + h / 2])
    elif lb.endswith(".xml"):  # xml file
        height, width = image_shape
        tree = parse(lb)
        root = tree.documentElement
        for child in root.getElementsByTagName("object"):
            name = child.getElementsByTagName("name")[0].childNodes[0].data.strip().lower()
            xtl, ytl, xbr, ybr = float("inf"), float("inf"), -1, -1
            for pt in child.getElementsByTagName("pt"):
                x = float(pt.getElementsByTagName("x")[0].childNodes[0].data)
                y = float(pt.getElementsByTagName("y")[0].childNodes[0].data)
                xtl = min(xtl, x)
                ytl = min(ytl, y)
                xbr = max(xbr, x)
                ybr = max(ybr, y)
            labels.append([name, xtl / width, ytl / height, xbr / width, ybr / height])
    return labels


def travel_files(queue, class_names=None, label_suffix=".txt", img_dir="/images", label_dir="/labels"):
    """
    1~9: jump to n% of dataset
    a: prev image
    *: next image
    q: quit
    """
    # control loop
    i = 0
    canvas = Canvas()
    while i < len(queue):
        fp = queue[i]
        print(fp)
        image = cv2.imread(fp)
        image = im2tensor(image).unsqueeze(0)
        height, width = image.shape[-2:]
        lb = fp.replace(".png", label_suffix).replace(".jpg", label_suffix).replace(img_dir, label_dir)
        lb = read_labels_as_xyxy(lb, (height, width), class_names)
        canvas.load(image[0])
        for title, x1, y1, x2, y2 in lb:
            pt1, pt2 = (x1 * width, y1 * height), (x2 * width, y2 * height)
            color = canvas.color(title)
            canvas.draw_box(pt1, pt2, alpha=0.4, thickness=-1, color=color)
            canvas.draw_box(pt1, pt2, color=color, title=title)
        canvas.show("ground truth")
        flag = cv2.waitKey(0)
        if flag == ord("q"):
            break
        elif flag == ord("a"):
            i -= 1
        elif flag in [ord(x) for x in "1234567890"]:
            i = int(len(queue) * (flag - ord("0")) / 10)
        else:
            i += 1


def travel_dataset(path, class_names=None, label_suffix=".txt", img_dir="/images", label_dir="/labels"):
    """
    1~9: jump to n% of dataset
    a: prev image
    *: next image
    q: quit
    """
    # load image paths in queue
    dataset_queue = []
    for image in os.listdir(path):
        if not image.startswith("._") and (image.endswith(".png") or image.endswith(".jpg")):
            fp = f"{path}/{image}"
            dataset_queue.append(fp)
    travel_files(dataset_queue, class_names, label_suffix, img_dir, label_dir)
