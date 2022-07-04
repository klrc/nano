import os
from xml.dom.minidom import parse
import cv2


coco_class_names = "person|bicycle|car|motorbike|airplane|bus|train|truck|boat|traffic light|fire hydrant|stop sign|parking meter|bench|bird|cat|dog|horse|sheep|cow|elephant|bear|zebra|giraffe|backpack|umbrella|handbag|tie|suitcase|frisbee|skis|snowboard|sports ball|kite|baseball bat|baseball glove|skateboard|surfboard|tennis racket|bottle|wine glass|cup|fork|knife|spoon|bowl|banana|apple|sandwich|orange|broccoli|carrot|hot dog|pizza|donut|cake|chair|couch|potted plant|bed|table|toilet|monitor|laptop|mouse|remote|keyboard|phone|microwave|oven|toaster|sink|fridge|book|clock|vase|scissors|toy|hair drier|toothbrush|gun|person hand".split(  # noqa:E501
    "|"
)  # class names


def strike(text):
    result = ""
    for c in text:
        result = result + c + "\u0336"
    return result


def load_labels_as_xyxy(x, y, w, h, height, width):
    pt1 = int((x - w / 2) * width), int((y - h / 2) * height)
    pt2 = int((x + w / 2) * width), int((y + h / 2) * height)
    return pt1, pt2


def search_in_coco_classes(name):
    for coco_name in coco_class_names:
        if (coco_name == name) or (coco_name in name.split(" ")):
            return True, coco_name
    return False, name


def replace_label(name, alias, target):
    if (alias == name) or (alias in name.replace(",", " ").split(" ")):
        return target
    return name


def process(debug=False):

    rules = []

    image_root = "../datasets/IndoorCVPR_09/ImagesRaw"
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

            # if not well-formed (invalid token): line 844, column 5 occurs, add this line then edit
            # print(xml_path)

            tree = parse(xml_path)
            root = tree.documentElement
            export_data = []

            for child in root.getElementsByTagName("object"):
                name = child.getElementsByTagName("name")[0].childNodes[0].data.strip().lower()
                raw_name = name
                # fix typo
                name = replace_label(name, "television set", "television_set")
                name = replace_label(name, "television stand", "television_stand")
                name = replace_label(name, "television case", "television_case")
                name = replace_label(name, "toys", "toy")
                name = replace_label(name, "toy car", "toy")
                name = replace_label(name, "toilet seat lid", "toilet_seat_lid")
                name = replace_label(name, "toilet bowl", "toilet")
                name = replace_label(name, "toilet seat", "toilet")
                name = replace_label(name, "sandwichs", "sandwich")
                name = replace_label(name, "personsitting", "person")
                name = replace_label(name, "peson walking", "person")
                name = replace_label(name, "tv", "monitor")
                name = replace_label(name, "motorcycle", "motorbike")
                name = replace_label(name, "chairs", "chair")
                name = replace_label(name, "man", "person")
                name = replace_label(name, "woman", "person")
                name = replace_label(name, "perso", "person")
                name = replace_label(name, "books", "book")
                name = replace_label(name, "tv stand", "tv_stand")
                name = replace_label(name, "desk", "table")
                name = replace_label(name, "table lamp", "table_lamp")
                name = replace_label(name, "toilet paper dispenser", "toilet_paper")
                name = replace_label(name, "toilet paper", "toilet_paper")
                name = replace_label(name, "wc", "toilet")
                name = replace_label(name, "lying down", "person")
                name = replace_label(name, "human", "person")
                name = replace_label(name, "plant pot occluded", "potted plant")
                name = replace_label(name, "plant pot", "potted plant")
                name = replace_label(name, "plant in a pot", "potted plant")
                name = replace_label(name, "van", "car")
                name = replace_label(name, "television", "monitor")
                name = replace_label(name, "televison", "monitor")
                name = replace_label(name, "seat", "chair")
                name = replace_label(name, "seats", "bench")
                name = replace_label(name, "sofa", "couch")
                name = replace_label(name, "key board", "keyboard")
                name = replace_label(name, "fridger occluded", "fridge")
                name = replace_label(name, "refrigerator", "fridge")
                name = replace_label(name, "refigetator", "fridge")
                name = replace_label(name, "refrigator", "fridge")
                name = replace_label(name, "bottl", "bottle")
                is_matched, name = search_in_coco_classes(name)

                # collect labels for further process
                if is_matched:
                    rule = f"{raw_name} -> {name}"
                else:
                    rule = strike(raw_name)
                if rule not in rules:
                    rules.append(rule)
                    print(rule)

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

            unique_name = fname.replace(".jpg", "").replace(" ", "_")
            unique_name = f"{scene}_{unique_name}"

            if not debug:
                # os.system(f'cp "{image_path}" {generated_image_root}/{unique_name}.jpg')
                with open(f"{generated_label_root}/{unique_name}.txt", "w") as f:
                    for line in export_data:
                        line = " ".join([str(x) for x in line])
                        f.write(f"{line}\n")
                print(f"{generated_image_root}/{unique_name}.jpg")


if __name__ == "__main__":
    process(debug=False)
