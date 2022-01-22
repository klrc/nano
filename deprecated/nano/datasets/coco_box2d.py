import os
import cv2
import numpy as np
import torch
import math
from tqdm import tqdm

from ._layer import DatasetLayer
from nano.ops import xywh2xyxy


class MSCOCO(DatasetLayer):
    """
    dataset loader for yolov5-ultralytics format.
    should specify:
        imgs_root which contains all *.jpg
        annotations_root which contains all *.txt
    annotations should be formatted as:
        class_id, x, y, w, h
    """

    def __init__(self, imgs_root=None, annotations_root=None, min_size=None, max_size=None, class_map=None) -> None:
        super().__init__()
        self.data = []
        # load & check annotation exists
        if imgs_root is not None and annotations_root is not None:
            corrupted = 0
            for img in tqdm(os.listdir(imgs_root), desc=f"verifying data from {imgs_root}"):
                iid, suffix = img.split(".")
                image_path = f"{imgs_root}/{iid}.{suffix}"
                annotation_path = f"{annotations_root}/{iid}.txt"
                if self.verify(image_path, annotation_path):
                    labels = self.cache_label(annotation_path, class_map)
                    self.data.append((image_path, labels))
                else:
                    corrupted += 1
            if corrupted > 0:
                print(f'Warning: Ignored {corrupted} corrupted data.')
        # Cache labels into memory for faster training
        assert min_size is None or max_size is None, "only 1 of min/max can be set"
        self.min_size = min_size
        self.max_size = max_size

    def __add__(self, other):
        assert isinstance(other, MSCOCO)
        new_dataset = MSCOCO(min_size=None, max_size=None)
        new_dataset.data = self.data + other.data
        return new_dataset

    def verify(self, image_path, annotation_path):
        # check whether image & annotation is valid
        # self.log(f"verifying {image_path}")
        try:
            # Image.open(image_path).verify()
            assert os.path.exists(annotation_path), annotation_path
            with open(annotation_path, "r") as f:
                for x in f.readlines():
                    assert len(x.strip().split(" ")) == 5, x
        except Exception:
            # print(f"WARNING: Ignoring corrupted label of image {image_path}")
            return False
        return True

    def cache_label(self, annotation_path, class_map):
        """
        yield label only
        """
        ret = []
        with open(annotation_path, "r") as f:
            if class_map is None:
                for x in f.readlines():
                    ret.append(x.strip().split(" "))
            else:
                for x in f.readlines():
                    lb = x.strip().split(" ")
                    cid = int(lb[0])
                    if cid in class_map:
                        lb[0] = class_map[cid]
                        ret.append(lb)
        return tuple(ret)

    def __getitem__(self, index):
        """
        returns:
            image: (h, w, bgr)
            labels: (N, cxyxy)
        """
        # get image & labels for sample i
        # load image & resize
        image_path, labels = self.data[index]
        img = cv2.imread(image_path)  # BGR
        height, width = img.shape[:2]  # orig hw
        assert height * width > 0
        if self.min_size is not None:
            ratio = self.min_size / min(height, width)
        elif self.max_size is not None:
            ratio = self.max_size / max(height, width)
        else:
            ratio = 1
        if ratio != 1:
            img = cv2.resize(
                img,
                (int(width * ratio), int(height * ratio)),
                interpolation=cv2.INTER_AREA,
            )
        # load annotations
        transformed_labels = []
        for c, x, y, w, h in labels:
            transformed_labels.append(
                [
                    int(c),
                    float(x) * width * ratio,
                    float(y) * height * ratio,
                    float(w) * width * ratio,
                    float(h) * height * ratio,
                ]
            )
        labels = np.array(transformed_labels)
        # normalized xywh to pixel xyxy format
        if len(labels) > 0:
            labels[:, 1:] = xywh2xyxy(labels[:, 1:])
        labels = labels.copy()
        return img, labels

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    """
    collate function for torch.utils.dataloader()
    e.g.
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            num_workers=8,
            pin_memory=False,
            collate_fn=collate_fn,  <---- set here
        )
    """
    img_batch, label_batch = zip(*batch)  # transposed
    targets = []
    for i, l in enumerate(label_batch):
        li = torch.ones((l.size(0), 1)) * i  # add target image index for build_targets()
        targets.append(torch.cat([li, l], dim=1))
    return torch.stack(img_batch, 0), torch.cat(targets, 0)


def letterbox_collate_fn(batch):
    """
    letterbox collate function for torch.utils.dataloader()
    pad inputs & labels to max-size(32x) of batch
    label: N - <c, x, y, x, y>
    """

    img_batch, _ = zip(*batch)  # transposed
    max_h = math.ceil(max([x.shape[-2] for x in img_batch]) / 32) * 32
    max_w = math.ceil(max([x.shape[-1] for x in img_batch]) / 32) * 32
    img_batch = torch.ones((len(img_batch), img_batch[0].size(0), max_h, max_w)) * 114
    targets = []
    for i, (img, _l) in enumerate(batch):
        _, h, w = img.shape
        pad_h = (max_h - h) // 2
        pad_w = (max_w - w) // 2
        img_batch[i, :, pad_h: pad_h + h, pad_w: pad_w + w] = img
        pads = torch.tensor([0, pad_w, pad_h, pad_w, pad_h])
        _l = _l + pads  # after padding
        li = torch.ones((_l.size(0), 1)) * i  # add target image index for build_targets()
        targets.append(torch.cat([li, _l], dim=1))
    return img_batch, torch.cat(targets, 0)
