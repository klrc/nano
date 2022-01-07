import numpy as np
import torch
import random
import cv2
import albumentations as A
from tqdm import tqdm

from .coco_box2d_visualize import from_numpy_image
from .coco_box2d import MSCOCO
from ._layer import DatasetLayer


class Mosaic4(DatasetLayer):
    """
    4-mosaic augmentation from ultralytics-yolov5
    """

    def __init__(self, base, img_size=416, loss_thres=0.8) -> None:
        super().__init__(base, DatasetLayer)
        self.img_size = img_size
        self.loss_thres = loss_thres

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        # prepare mosaic center divider as (0.5-1.5x, 0.5-1.5x) in (2x, 2x)
        s = self.img_size
        # yc = int(np.clip(np.random.normal(s, 16), s // 2, 2 * s - s // 2))
        # xc = int(np.clip(np.random.normal(s, 16), s // 2, 2 * s - s // 2))
        yc = int(random.uniform(s // 2, 2 * s - s // 2))
        xc = int(random.uniform(s // 2, 2 * s - s // 2))
        # loads images in a 4-mosaic
        # with 3 additional image indices
        indices = [index] + random.choices([x for x in range(len(self))], k=3)
        # base image with 4 tiles
        img4 = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        labels4 = []

        for i, ind in enumerate(indices):
            img, labels = self.base.__getitem__(ind)
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
            labels = labels.copy()
            if labels.size > 0:
                # add offset
                labels[:, 1] += offset_x
                labels[:, 3] += offset_x
                labels[:, 2] += offset_y
                labels[:, 4] += offset_y
                labels4.append(labels)

        # crop black border to average
        border = int(s // 2)
        img4 = img4[border : 2 * s - border, border : 2 * s - border]
        if len(labels4) > 0:
            # Concat/clip labels
            labels4 = np.concatenate(labels4, 0)
            raw_boxsize = (labels4[:, 3] - labels4[:, 1]) * (labels4[:, 4] - labels4[:, 2])
            labels4[:, 1:] = np.clip(labels4[:, 1:] - border, 0, s - 1)
            new_boxsize = (labels4[:, 3] - labels4[:, 1]) * (labels4[:, 4] - labels4[:, 2])
            keeped_boxes = new_boxsize / (raw_boxsize + 1e-8) > (1 - self.loss_thres)
            labels4 = labels4[keeped_boxes]
        return img4, labels4


class Affine(DatasetLayer):
    """
    Affine augmentation for object detection dataset
    including:
        - horizontal flip
        - perspective
    """

    def __init__(self, base, horizontal_flip=0.5, perspective=0.5, max_perspective=0.15, crop_border=True, loss_thres=0.1) -> None:
        super().__init__(base, DatasetLayer)
        self.flip = horizontal_flip
        self.perspective = perspective
        self.max_perspective = max_perspective
        self.crop_border = crop_border
        self.loss_thres = loss_thres

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        img, label = self.base.__getitem__(index)
        h, w, _ = img.shape
        # random horizontal flip
        if random.random() < self.flip:
            img = img[:, ::-1]
            if len(label) > 0:
                new_x1 = w - label[:, 3]
                new_x2 = w - label[:, 1]
                label[:, 1] = new_x1
                label[:, 3] = new_x2
        # perspective
        if random.random() < self.perspective:

            def rand_offset():
                return random.random() * self.max_perspective

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
            img = cv2.warpPerspective(img, M, (new_w, new_h))
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
                border_h = (new_h - h) // 2
                border_w = (new_w - w) // 2
                img = img[border_h : new_h - border_h, border_w : new_w - border_w]
                if len(label) > 0:
                    raw_boxsize = (label[:, 3] - label[:, 1]) * (label[:, 4] - label[:, 2])
                    new_boxsize = (label[:, 3] - label[:, 1]) * (label[:, 4] - label[:, 2])
                    label[:, 1] = np.clip(label[:, 1] - border_w, 0, w - 1)
                    label[:, 2] = np.clip(label[:, 2] - border_h, 0, h - 1)
                    label[:, 3] = np.clip(label[:, 3] - border_w, 0, w - 1)
                    label[:, 4] = np.clip(label[:, 4] - border_h, 0, h - 1)
                    keeped_boxes = new_boxsize / (raw_boxsize + 1e-8) > (1 - self.loss_thres)
                    label = label[keeped_boxes]
        return img, label


class Albumentations(DatasetLayer):
    """
    basic image-only augmentations from albumentations
    """

    def __init__(self, base, transforms="random_blind") -> None:
        super().__init__(base, DatasetLayer)
        if transforms == "random_blind":
            transforms = [
                A.Blur(p=0.05),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.1),
            ]
        self.transforms = A.Compose(transforms)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        img, label = self.base.__getitem__(index)
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
        return img, label


class ClassMapping(DatasetLayer):
    """
    map dataset label classes
    """

    def __init__(self, base=None, map_dict=None) -> None:
        super().__init__(base, MSCOCO)
        assert isinstance(map_dict, dict)
        self.mapping = map_dict
        self.data = self.base.data

    def __len__(self):
        return len(self.base)

    def _yield_labels(self, index):
        for lb in self.base._yield_labels(index):
            cid = int(lb[0])
            if cid in self.mapping:
                lb[0] = self.mapping[cid]
                yield lb


    def __getitem__(self, index):
        img, labels = self.base.__getitem__(index)
        new_labels = []
        for lb in labels:
            cid = lb[0]
            if cid in self.mapping:
                lb[0] = self.mapping[cid]
                new_labels.append(lb)
        new_labels = np.array(new_labels)
        return img, new_labels


class SizeLimit(DatasetLayer):
    """
    limit the dataset size for quick scratch-training
    """

    def __init__(self, base, limit=5000, sort_only=False) -> None:
        super().__init__(base, MSCOCO, ClassMapping)
        if sort_only:
            limit = len(self.base)
        self.limit = min(limit, len(self.base))
        self.data = []

    def __len__(self):
        return self.limit

    def __getitem__(self, index):
        if len(self.data) == 0:
            # for i in range(len(self.base)):
            for i in tqdm(range(len(self.base)), desc="prepare for size limit"):
                box_cids = []
                for c, _, _, _, _ in self.base._yield_labels(i):
                        box_cids.append(c)
                diversity = len(set(box_cids))
                self.data.append((i, diversity))
            self.data = sorted(self.data, key=lambda x: x[1], reverse=True)
            self.data = [x[0] for x in self.data][: self.limit]
        img, label = self.base.__getitem__(self.data[index])
        return img, label


class ToTensor(DatasetLayer):
    """
    convert numpy to pytorch tensor.
    """

    def __init__(self, base) -> None:
        super().__init__(base, DatasetLayer)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        """
        normalize image & labels to pytorch tensors
        returns:
            image: (rgb, h, w) (normalized)
            target: (N, cxyxy)
        """
        image, labels = self.base.__getitem__(index)
        tensor_im = from_numpy_image(image)
        # process label into (cid, x, y, x, y) (torch.from_numpy())
        labels = [x for x in labels if (x[3] - x[1]) > 2 and (x[4] - x[2]) > 2]  # valid label
        target = torch.zeros((len(labels), 5))
        for i, label in enumerate(labels):
            target[i] += torch.from_numpy(label)
        return tensor_im, target
