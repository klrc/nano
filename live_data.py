import os
import random
import cv2
import numpy as np
import torch
from utils import from_numpy_image


def file_io_layer(root):
    # extract files from specified root
    if os.path.exists(root):
        for fname in os.listdir(root):
            yield fname, f"{root}/{fname}"


def detection_data_layer(*roots):
    # generate detction dataset items
    unpaired = {}
    for root in roots:
        for name, path in file_io_layer(root):
            name, _ = name.split(".")
            if name in unpaired:
                yield unpaired.pop(name), path
            else:
                unpaired[name] = path


def infinite_sampler(*data_generators, shuffle=True):
    # infinite sampler from finite data generators
    data = []
    for generator in data_generators:
        data += [x for x in generator]
    if shuffle:
        while True:
            yield random.choice(data)
    else:
        while True:
            for x in data:
                yield x


def nparray_layer(data_generator):
    # unify data format from detection_data_layer
    for data in data_generator:
        image, label = sorted(data, key=lambda x: x.split(".")[-1])
        image = cv2.imread(image)  # BGR
        image_size_factor = image.shape[:2][::-1]  # orig hw
        # process label
        # relative cxywh -> absolute cxyxy
        with open(label, "r") as f:
            label = []
            for line in f.readlines():
                line = np.array([float(x) for x in line.split(" ")])
                line[1:3] *= image_size_factor
                line[3:] *= image_size_factor
                line[1:3] -= line[3:] / 2
                line[3:]  += line[1:3]
                label.append(line)
        label = np.array(label)
        yield image, label


def hsv_transform(data_generator, hgain=0.5, sgain=0.5, vgain=0.5, p=1):
    # HSV color-space augmentation
    for image, label in data_generator:
        if random.random() < p:
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
            dtype = image.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            image = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        yield image, label


def to_tensor(data_generator):
    """
    normalize image & labels to pytorch tensors
    returns:
        image: (rgb, h, w) (normalized)
        target: (N, cxyxy)
    """
    for image, labels in data_generator:
        tensor_im = from_numpy_image(image)
        # process label into (cid, x, y, x, y) (torch.from_numpy())
        labels = [x for x in labels if (x[3] - x[1]) > 2 and (x[4] - x[2]) > 2]  # valid label
        target = torch.zeros((len(labels), 5))
        for i, label in enumerate(labels):
            target[i] += torch.from_numpy(label)
        yield tensor_im, target


if __name__ == "__main__":
    dataset1 = detection_data_layer("../datasets/VOC/images/train2012", "../datasets/VOC/labels/train2012")
    dataset2 = detection_data_layer("../datasets/VOC/images/val2012", "../datasets/VOC/labels/val2012")
    layer = infinite_sampler(dataset1, dataset2, shuffle=True)
    layer = nparray_layer(layer)
    layer = hsv_transform(layer)
    layer = to_tensor(layer)

    from utils import voc_classes, draw_bounding_boxes

    for i, (image, labels) in enumerate(layer):
        str_labels = [voc_classes[int(x)] for x in labels[..., 0]]
        cv_img = draw_bounding_boxes(image, boxes=labels[..., 1:], boxes_label=str_labels)
        cv2.imwrite(f"test_live_data_{i}.png", cv_img)
        if i > 10:
            break
