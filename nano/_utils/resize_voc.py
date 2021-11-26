import os
import xml.etree.ElementTree as ET
import cv2
from tqdm import tqdm
import shutil


target_size = [224, 416]
target_root = "/home/sh/Projects/klrc/nano/nano/_utils/xnnc/Example/yolox-series/dataset"
if os.path.exists(target_root):
    shutil.rmtree(target_root)
os.makedirs(target_root)
with open(f"{target_root}/imagelist.txt", "a") as f:
    f.write("100\n")
    f.write("detection_out:detection_out\n")
with open(f"{target_root}/voc2012_labels.txt", "w") as f:
    f.write("background\n")
    f.write("person\n")
    f.write("bike\n")
    f.write("car\n")
target_class_mapping = {
    "person": "person",
    "bicycle": "bike",
    "motorcycle": "bike",
    "car": "car",
    "bus": "car",
}


root = "/home/sh/Datasets/VOC/images"
for img_file in tqdm(os.listdir(root + "/val2012")):
    img_path = root + "/val2012/" + img_file
    xml_root = root + "/VOCdevkit/VOC2012/Annotations"
    xml_path = xml_root + "/" + img_file.replace(".jpg", ".xml")

    # letterbox
    img = cv2.imread(img_path)
    raw_shape = img.shape
    scale_ratio = min(target_size[0] / raw_shape[0], target_size[1] / raw_shape[1])
    img = cv2.resize(img, (int(raw_shape[1] * scale_ratio), int(raw_shape[0] * scale_ratio)))
    cv2.imwrite("tmp.png", img)
    new_shape = img.shape
    paddings = (target_size[0] - new_shape[0], target_size[1] - new_shape[1])
    paddings = (int(paddings[0] / 2), int(paddings[1] / 2))
    img = cv2.copyMakeBorder(
        img, paddings[0], paddings[0], paddings[1], paddings[1], cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # parse xml
    tree = ET.parse(xml_path)
    tree_root = tree.getroot()
    for rm in [obj for obj in tree_root.findall("object") if obj.find("name").text not in target_class_mapping]:
        tree_root.remove(rm)
        
    tree_root.find("size").find("width").text = str(target_size[1])
    tree_root.find("size").find("height").text = str(target_size[0])
    for obj in tree_root.findall("object"):
        obj.find("name").text = target_class_mapping[obj.find("name").text]
        xmin_xml = obj.find("bndbox").find("xmin")
        ymin_xml = obj.find("bndbox").find("ymin")
        xmax_xml = obj.find("bndbox").find("xmax")
        ymax_xml = obj.find("bndbox").find("ymax")
        axis_data = [int(xmin_xml.text), int(ymin_xml.text), int(xmax_xml.text), int(ymax_xml.text)]
        axis_data = [int(x * scale_ratio) for x in axis_data]
        # paddings: vertical, horizontal
        # axis_data: x1, y1, x2, y2
        axis_data[0] += paddings[1]
        axis_data[2] += paddings[1]
        axis_data[1] += paddings[0]
        axis_data[3] += paddings[0]
        # for test ------------------------
        # x1, y1, x2, y2 = axis_data
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (75, 0, 130), 1, cv2.LINE_AA, 0)
        # cv2.imwrite("tmp.png", img)
        xmin_xml.text = str(axis_data[0])
        ymin_xml.text = str(axis_data[1])
        xmax_xml.text = str(axis_data[2])
        ymax_xml.text = str(axis_data[3])

    cv2.imwrite(f"{target_root}/{img_file}", img)
    tree.write(f'{target_root}/{img_file.replace(".jpg",".xml")}', encoding="utf-8")
    with open(f"{target_root}/imagelist.txt", "a") as f:
        f.write(f"{img_file}\n")
