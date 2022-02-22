import random
from tkinter.messagebox import NO
import cv2
import numpy as np
import torch
import math


class TransformFunction:
    feed_samples = 1

    def __init__(self, p=1) -> None:
        self.p = p

    def __call__(self, data):
        raise NotImplementedError


class Numpy2Image(TransformFunction):
    def __init__(self) -> None:
        super().__init__(p=1)

    def __call__(self, data):
        image, label = data
        return self.functional(image), label

    @staticmethod
    def functional(image):
        """
        converts numpy image to tensor
        (inplace-safe function)
        returns:
            x: (rgb, h, w)
        """
        x = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        x = np.ascontiguousarray(x)
        x = torch.from_numpy(x).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
        return x


class Tensor2Image(TransformFunction):
    def __init__(self) -> None:
        super().__init__(p=1)

    def __call__(self, data):
        image, label = data
        return self.functional(image), label

    @staticmethod
    def functional(x):
        """
        converts tensor to numpy image
        returns:
            np_img: (h, w, bgr)
        """
        np_img = (x * 255.0).int().numpy()
        np_img = np_img[::-1].transpose((1, 2, 0))  # CHW to HWC, RGB to BGR, 0~1 to 0~255
        np_img = np.ascontiguousarray(np_img)
        return np_img


class IndexMapping(TransformFunction):
    def __init__(self, index_map) -> None:
        super().__init__(p=1)
        self.index_map = index_map

    def __call__(self, data):
        image, label = data  # non-match mode
        # ready to process
        new_label = []
        for lb in label:
            cid = int(lb[0])
            if cid in self.index_map:
                lb[0] = self.index_map[cid]
                new_label.append(lb)
        label = np.array(new_label)
        return image, label


class HSVTransform(TransformFunction):
    def __init__(self, hgain=0.015, sgain=0.7, vgain=0.4, p=1) -> None:
        super().__init__(p)
        # hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
        # hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
        # hsv_v: 0.4  # image HSV-Value augmentation (fraction)
        # HSV color-space augmentation
        self.gains = (hgain, sgain, vgain)

    def __call__(self, data):
        image, label = data
        r = np.random.uniform(-1, 1, 3) * self.gains + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        dtype = image.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        return image, label


class CenterCrop(TransformFunction):
    def __init__(self, cropped_size, min_iou=0.1, p=1) -> None:
        super().__init__(p)
        self.cropped_size = cropped_size
        self.min_iou = min_iou

    def __call__(self, data):
        """
        perform center crop on img & label
        warning: inplace operations in this function
        """
        image, label = data
        return self.functional(image, label, self.cropped_size, self.min_iou)

    @staticmethod
    def functional(image, label, cropped_size, min_iou):
        h, w, _ = image.shape
        ch, cw = cropped_size
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


class CenterPad(TransformFunction):
    def __init__(self, padded_size, pad_value=114, p=1) -> None:
        super().__init__(p)
        self.padded_size = padded_size
        self.pad_value = pad_value

    def __call__(self, data):
        """
        perform center padding on img & label
        warning: inplace operations in this function
        """
        image, label = data
        return self.functional(image, label, self.padded_size, self.pad_value)

    @staticmethod
    def functional(image, label, padded_size, pad_value):
        h, w, _ = image.shape
        ph, pw = padded_size
        border_h = (ph - h) // 2
        border_w = (pw - w) // 2
        new_img = np.ones((ph, pw, image.shape[-1]), image.dtype) * pad_value
        new_img[border_h : h + border_h, border_w : w + border_w] = image
        if len(label) > 0:
            # xyxy
            label[:, 1] += border_w
            label[:, 2] += border_h
            label[:, 3] += border_w
            label[:, 4] += border_h
        return new_img, label


class HorizontalFlip(TransformFunction):
    def __init__(self, p=1) -> None:
        super().__init__(p)

    def __call__(self, data):
        """
        basic horizontal flip function
        """
        image, label = data
        h, w, _ = image.shape
        image = image[:, ::-1]
        if len(label) > 0:
            new_x1 = w - label[:, 3]
            new_x2 = w - label[:, 1]
            label[:, 1] = new_x1
            label[:, 3] = new_x2
        return image, label


class Resize(TransformFunction):
    def __init__(self, ratio=None, min_size=None, max_size=None, p=1) -> None:
        super().__init__(p)
        self.ratio = ratio
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, data):
        image, label = data
        h, w, _ = image.shape
        if self.ratio is None:
            if self.min_size is not None:
                ratio = self.min_size / min(h, w)  # min image size -> align_min
            else:
                ratio = self.max_size / max(h, w)  # max image size -> align_max
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


class RandomScale(TransformFunction):
    def __init__(self, min_scale, max_scale, crop_border=False, min_iou=0.1, pad_value=114, p=1) -> None:
        super().__init__(p)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.crop_border = crop_border
        self.min_iou = min_iou
        self.pad_value = pad_value

    def __call__(self, data):
        """
        random scale within range (max_scale, min_scale)
        """
        image, label = data
        h, w, _ = image.shape
        ratio = random.random() * (self.max_scale - self.min_scale) + self.min_scale
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
        if self.crop_border:
            if ratio > 1:
                image, label = CenterCrop.functional(image, label, (h, w), self.min_iou)
            # elif ratio < 1:
            #     image, label = CenterPad.functional(image, label, (h, w), self.pad_value)
        return image, label


class RandomAffine(TransformFunction):
    def __init__(self, min_scale, max_scale, crop_border=True, min_iou=0.1, pad_value=114, p=1) -> None:
        super().__init__(p)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.crop_border = crop_border
        self.min_iou = min_iou
        self.pad_value = pad_value

    def __call__(self, data):
        """
        random perspective affine transform function
        """
        image, label = data

        def rand_offset():
            return random.random() * (self.max_scale - self.min_scale) + self.min_scale - 1

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
        image = cv2.warpPerspective(image, M, (new_w, new_h), borderValue=(self.pad_value,) * 3)
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
        if self.crop_border:
            image, label = CenterCrop.functional(image, label, (h, w), self.min_iou)
        return image, label


class Mosaic4(TransformFunction):
    feed_samples = 4

    def __init__(self, mosaic_size, pad_value=114, min_iou=0.1, p=1) -> None:
        super().__init__(p)
        self.mosaic_size = mosaic_size
        self.pad_value = pad_value
        self.min_iou = min_iou

    def __call__(self, data):
        """
        4-mosaic augmentation from ultralytics-yolov5
        loads images in a 4-mosaic
        """
        # prepare mosaic center divider as (0.5-1.5x, 0.5-1.5x) in (2x, 2x)
        s = self.mosaic_size
        # yc = int(np.clip(np.random.normal(s, 16), s // 2, 2 * s - s // 2))
        # xc = int(np.clip(np.random.normal(s, 16), s // 2, 2 * s - s // 2))
        yc = int(random.uniform(s // 2, 2 * s - s // 2))
        xc = int(random.uniform(s // 2, 2 * s - s // 2))
        # base image with 4 tiles
        img4 = np.full((s * 2, s * 2, 3), self.pad_value, dtype=np.uint8)
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
        img4, labels4 = CenterCrop.functional(img4, labels4, (s, s), self.min_iou)
        return img4, labels4


class AlbumentationsPreset(TransformFunction):
    def __init__(self, preset="random_blind", p=1) -> None:
        super().__init__(p)
        import albumentations as A

        if preset == "random_blind":
            self.transforms = A.Compose(
                [
                    A.Blur(p=0.01),
                    A.MedianBlur(p=0.01),
                    A.ToGray(p=0.01),
                ]
            )
        else:
            raise NotImplementedError

    def __call__(self, data):
        image, label = data
        image = self.transforms(image=image)["image"]
        return image, label


class ToTensor(TransformFunction):
    def __init__(self, p=1) -> None:
        super().__init__(p)

    def __call__(self, data):
        """
        normalize image & labels to pytorch tensors
        returns:
            image: (rgb, h, w)  raw
            target: (cxyxy)     nx5
        """
        image, labels = data
        tensor_im = Numpy2Image.functional(image)
        # process label into (cid, x, y, x, y) (torch.from_numpy())
        labels = [x for x in labels if (x[3] - x[1]) > 2 and (x[4] - x[2]) > 2]  # valid label
        target = torch.zeros((len(labels), 5))
        for i, label in enumerate(labels):
            target[i] += torch.from_numpy(label)
        return tensor_im, target


def letterbox_collate_fn(batch, max_stride=32):
    """
    letterbox collate function for torch.utils.dataloader()
    pad inputs & labels to max-size(32x) of batch
    returns:
        image: (N, C, H, W)     normalized
        label: (N, cxyxy)       nx6
    """

    img_batch, _ = zip(*batch)  # transposed
    max_h = math.ceil(max([x.shape[-2] for x in img_batch]) / max_stride) * max_stride
    max_w = math.ceil(max([x.shape[-1] for x in img_batch]) / max_stride) * max_stride
    img_batch = torch.ones((len(img_batch), img_batch[0].size(0), max_h, max_w)) * 0.5
    targets = []
    for i, (img, _l) in enumerate(batch):
        _, h, w = img.shape
        pad_h = (max_h - h) // 2
        pad_w = (max_w - w) // 2
        img_batch[i, :, pad_h : pad_h + h, pad_w : pad_w + w] = img
        pads = torch.tensor([0, pad_w, pad_h, pad_w, pad_h])
        _l = _l + pads  # after padding
        li = torch.ones((_l.size(0), 1)) * i  # add target image index for build_targets()
        targets.append(torch.cat([li, _l], dim=1))
    return img_batch, torch.cat(targets, 0)
