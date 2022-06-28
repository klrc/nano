import os
from xml.dom.minidom import parse
import cv2
import sys

sys.path.append(".")
from test_utils.detection_visualize import Canvas


coco_class_names = "people|bicycle|car|motorbike|airplane|bus|train|truck|boat|traffic light|fire hydrant|stop sign|parking meter|bench|bird|cat|dog|horse|sheep|cow|elephant|bear|zebra|giraffe|backpack|umbrella|handbag|tie|suitcase|frisbee|skis|snowboard|sports ball|kite|baseball bat|baseball glove|skateboard|surfboard|tennis racket|bottle|wine glass|cup|fork|knife|spoon|bowl|banana|apple|sandwich|orange|broccoli|carrot|hot dog|pizza|donut|cake|chair|couch|potted plant|bed|table|toilet|tv|laptop|mouse|remote|keyboard|cell phone|microwave|oven|toaster|sink|refrigerator|book|clock|vase|scissors|teddy bear|hair drier|toothbrush|gun|person hand|toy".split(  # noqa:E501
    "|"
)  # class names


def load_labels_as_xyxy(x, y, w, h, height, width):
    pt1 = int((x - w / 2) * width), int((y - h / 2) * height)
    pt2 = int((x + w / 2) * width), int((y + h / 2) * height)
    return pt1, pt2


def search_in_coco_classes(name):
    for coco_name in coco_class_names:
        if coco_name in name:
            return True, coco_name
    return False, name


def replace_label(name, alias, target):
    if alias in name:
        return target
    return name


def process(vis=False):

    # class_names = []
    # unexpected_names = []

    image_root = "../datasets/IndoorCVPR_09/ImagesRaw"

    # create dir
    generated_image_root = "../datasets/IndoorCVPR_09/images"
    generated_label_root = "../datasets/IndoorCVPR_09/labels"
    for path in (generated_image_root, generated_label_root):
        if not os.path.exists(path):
            os.makedirs(path)

    for scene in os.listdir(image_root):
        if scene.startswith("._"):
            continue

        for fname in os.listdir(f"{image_root}/{scene}"):
            if fname.startswith("._"):
                continue
            image_path = f"{image_root}/{scene}/{fname}"
            xml_path = image_path.replace("/ImagesRaw", "/Annotations").replace(".jpg", ".xml")
            if not os.path.exists(xml_path):
                continue

            im = cv2.imread(image_path)
            if im is None:
                continue
            height, width, _ = im.shape
            if vis:
                canvas = Canvas(im)

            # if not well-formed (invalid token): line 844, column 5 occurs, add this line then edit
            # print(xml_path)

            tree = parse(xml_path)
            root = tree.documentElement
            export_data = []

            for child in root.getElementsByTagName("object"):
                name = child.getElementsByTagName("name")[0].childNodes[0].data.strip().lower()
                # fix typo
                name = replace_label(name, "human", "people")
                name = replace_label(name, "plant in a pot", "potted plant")
                name = replace_label(name, "person", "people")
                name = replace_label(name, "van", "car")
                name = replace_label(name, "televison", "tv")
                name = replace_label(name, "sofa", "couch")
                name = replace_label(name, "key board", "keyboard")
                name = replace_label(name, "refigetator", "refrigerator")
                name = replace_label(name, "refrigator", "refrigerator")
                name = replace_label(name, "bottl", "bottle")
                is_matched, name = search_in_coco_classes(name)

                # collect labels for further process
                # if is_matched:
                #     if name not in class_names:
                #         class_names.append(name)
                # else:
                #     if name not in unexpected_names:
                #         unexpected_names.append(name)

                if is_matched:
                    xtl, ytl, xbr, ybr = float("inf"), float("inf"), -1, -1
                    for pt in child.getElementsByTagName("pt"):
                        x = float(pt.getElementsByTagName("x")[0].childNodes[0].data)
                        y = float(pt.getElementsByTagName("y")[0].childNodes[0].data)
                        xtl = min(xtl, x)
                        ytl = min(ytl, y)
                        xbr = max(xbr, x)
                        ybr = max(ybr, y)
                    xtl /= width
                    ytl /= height
                    xbr /= width
                    ybr /= height
                    objx, objy, objw, objh = (xtl + xbr) / 2, (ytl + ybr) / 2, xbr - xtl, ybr - ytl

                    export_data.append([int(coco_class_names.index(name)), objx, objy, objw, objh])

                    if vis:
                        pt1, pt2 = load_labels_as_xyxy(objx, objy, objw, objh, height, width)
                        canvas.draw_box(pt1, pt2, title=name)

            unique_name = fname.replace(".jpg", "").replace(" ", "_")
            unique_name = f"{scene}_{unique_name}"
            with open(f"{generated_label_root}/{unique_name}.txt", "w") as f:
                for line in export_data:
                    line = " ".join([str(x) for x in line])
                    f.write(f"{line}\n")

            os.system(f"cp \"{image_path}\" {generated_image_root}/{unique_name}.jpg")
            print(f"{generated_image_root}/{unique_name}.jpg")

            if vis:
                canvas.show(wait_key=True)


if __name__ == "__main__":
    process(vis=False)
