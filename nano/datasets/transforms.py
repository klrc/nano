import random
from tkinter.messagebox import NO
import cv2
import numpy as np


def hsv_transform(image, label, hgain=0.015, sgain=0.7, vgain=0.4):
    # hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
    # hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
    # hsv_v: 0.4  # image HSV-Value augmentation (fraction)
    # HSV color-space augmentation
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    dtype = image.dtype  # uint8

    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    image = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
    return image, label


def center_crop(image, label, raw_size, croped_size, min_iou=0.2):
    """
    perform center crop on img & label
    warning: inplace operations in this function
    """
    h, w = raw_size
    ch, cw = croped_size
    border_h = (h - ch) // 2
    border_w = (w - cw) // 2
    image = image[border_h : h - border_h, border_w : w - border_w]
    if len(label) > 0:
        raw_boxsize = (label[:, 3] - label[:, 1]) * (label[:, 4] - label[:, 2])
        label[:, 1] = np.clip(label[:, 1] - border_w, 0, cw - 1)
        label[:, 2] = np.clip(label[:, 2] - border_h, 0, ch - 1)
        label[:, 3] = np.clip(label[:, 3] - border_w, 0, cw - 1)
        label[:, 4] = np.clip(label[:, 4] - border_h, 0, ch - 1)
        new_boxsize = (label[:, 3] - label[:, 1]) * (label[:, 4] - label[:, 2])
        keeped_boxes = new_boxsize / (raw_boxsize + 1e-8) > min_iou
        label = label[keeped_boxes]
    return image, label


def center_padding(img, label, raw_size, padded_size, pad_value=114):
    """
    perform center padding on img & label
    warning: inplace operations in this function
    """
    h, w = raw_size
    ph, pw = padded_size
    border_h = (ph - h) // 2
    border_w = (pw - w) // 2
    new_img = np.ones((ph, pw, img.shape[-1]), img.dtype) * pad_value
    new_img[border_h : h + border_h, border_w : w + border_w] = img
    if len(label) > 0:
        # xyxy
        label[:, 1] += border_w
        label[:, 2] += border_h
        label[:, 3] += border_w
        label[:, 4] += border_h
    return new_img, label


def horizontal_flip(image, label):
    """
    basic horizontal flip function
    """
    h, w, _ = image.shape
    image = image[:, ::-1]
    if len(label) > 0:
        new_x1 = w - label[:, 3]
        new_x2 = w - label[:, 1]
        label[:, 1] = new_x1
        label[:, 3] = new_x2
    return image, label


import torch


def resize(image, label, ratio=None, min_size=None, max_size=None):
    h, w, _ = image.shape
    if ratio is None:
        if min_size is not None:
            ratio = min_size / min(h, w)  # min image size -> align_min
        else:
            ratio = max_size / max(h, w)  # max image size -> align_max
    new_h = int(h * ratio)
    new_w = int(w * ratio)
    image = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )
    # xyxy label
    if len(label) > 0:
        label[:, 1:] *= ratio
    return image, label


def random_scale(image, label, min_scale, max_scale, crop_border=False, min_iou=0.2, pad_value=114):
    """
    random scale within range (max_scale, min_scale)
    """
    h, w, _ = image.shape
    ratio = random.random() * (max_scale - min_scale) + min_scale
    new_h = int(h * ratio)
    new_w = int(w * ratio)
    image = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )
    # xyxy label
    if len(label) > 0:
        label[:, 1:] *= ratio
    # center crop
    if crop_border:
        if ratio > 1:
            image, label = center_crop(image, label, (new_h, new_w), (h, w), min_iou)
        elif ratio < 1:
            image, label = center_padding(image, label, (new_h, new_w), (h, w), pad_value)
    return image, label


def random_affine(image, label, min_scale, max_scale, crop_border=True, min_iou=0.2, pad_value=114):
    """
    random perspective affine transform function
    """

    def rand_offset():
        return random.random() * (max_scale - min_scale) + min_scale - 1

    h, w, _ = image.shape
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    # set offset positive
    offsets = [
        [-rand_offset() * w, -rand_offset() * h],  # top-left
        [rand_offset() * w, -rand_offset() * h],  # top-right
        [-rand_offset() * w, rand_offset() * h],  # bottom-left
        [rand_offset() * w, rand_offset() * h],  # bottom-right
    ]
    horizontal_min = min([x[0] for x in offsets])
    vertical_min = min([x[1] for x in offsets])
    offsets = np.float32([[x - horizontal_min, y - vertical_min] for x, y in offsets])
    pts2 = pts1 + offsets
    # get perspective transform metric
    new_w = int(max([x[0] for x in pts2]))
    new_h = int(max([x[1] for x in pts2]))
    M = cv2.getPerspectiveTransform(pts1, pts2)
    image = cv2.warpPerspective(image, M, (new_w, new_h), borderValue=(pad_value, pad_value, pad_value))
    if len(label) > 0:
        affined_label = []
        for box in label:
            x1, y1, x2, y2 = box[1:]
            rect_pts = np.array(
                [[[x1, y1]], [[x2, y1]], [[x1, y2]], [[x2, y2]]],
                dtype=np.float32,
            )
            rect_pts = cv2.perspectiveTransform(rect_pts, M)
            affined_label.append(
                [
                    box[0],
                    min(rect_pts[:, :, 0])[0],
                    min(rect_pts[:, :, 1])[0],
                    max(rect_pts[:, :, 0])[0],
                    max(rect_pts[:, :, 1])[0],
                ]
            )
        label = np.array(affined_label)
    # crop black border to average
    if crop_border:
        image, label = center_crop(image, label, (new_h, new_w), (h, w), min_iou)
    return image, label

def mosaic4(data, mosaic_size, pad_value=114, min_iou=0.2):
    """
    4-mosaic augmentation from ultralytics-yolov5
    loads images in a 4-mosaic
    """
    # prepare mosaic center divider as (0.5-1.5x, 0.5-1.5x) in (2x, 2x)
    s = mosaic_size
    # yc = int(np.clip(np.random.normal(s, 16), s // 2, 2 * s - s // 2))
    # xc = int(np.clip(np.random.normal(s, 16), s // 2, 2 * s - s // 2))
    yc = int(random.uniform(s // 2, 2 * s - s // 2))
    xc = int(random.uniform(s // 2, 2 * s - s // 2))
    # base image with 4 tiles
    img4 = np.full((s * 2, s * 2, 3), pad_value, dtype=np.uint8)
    labels4 = []

    for i, (img, label) in enumerate(data):
        h, w, _ = img.shape
        # place img in img4
        # set top-left offset for each image
        # xmin, ymin, xmax, ymax (mosaic(a)/raw(b) image)
        if i == 0:  # top left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        # map pixels to mosaic image
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        offset_x = x1a - x1b  # offsets for bbox
        offset_y = y1a - y1b
        # add labels
        label = label.copy()
        if label.size > 0:
            # add offset
            label[:, 1] += offset_x
            label[:, 3] += offset_x
            label[:, 2] += offset_y
            label[:, 4] += offset_y
            labels4.append(label)

    # Concat/clip labels
    if len(labels4) > 0:
        labels4 = np.concatenate(labels4, 0)

    # crop black border to average
    img4, labels4 = center_crop(img4, labels4, (2 * s, 2 * s), (s, s), min_iou)
    return img4, labels4
