import os
import random
import cv2
import numpy as np
import torch
from nano.datasets.visualize import from_numpy_image
import math


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


def load_data_as_numpy(data_generator):
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
                line[3:] += line[1:3]
                label.append(line)
        label = np.array(label)
        yield image, label


def class_mapping_layer(data_generator, src_names, target_names, map_dict):
    mapping = {}
    for k, v in map_dict.items():
        src_cid = src_names.index(k)
        target_cid = target_names.index(v)
        mapping[src_cid] = target_cid
    for image, label in data_generator:
        for i, lb in enumerate(label):
            cid = int(lb[0])
            if cid in mapping:
                label[i, 0] = mapping[cid]
        yield image, label


def transform_layer(data_generator, transform_fn, p=1, feed_samples=1, **kwargs):
    if feed_samples > 1:
        data_buffer = []
        for image, label in data_generator:
            if len(data_buffer) > 0 or random.random() < p:  # feed cache
                data_buffer.append((image, label))
                if len(data_buffer) == feed_samples:
                    image, label = transform_fn(data_buffer, **kwargs)
                    data_buffer.clear()
                    yield image, label
    else:
        for image, label in data_generator:
            if random.random() < p:
                image, label = transform_fn(image, label, **kwargs)
            yield image, label


def albumentations_transform_layer(data_generator, preset="random_blind", p=1):
    import albumentations as A

    if preset == "random_blind":
        transforms = A.Compose(
            [
                A.Blur(p=0.05),
                A.MedianBlur(p=0.05),
                A.ToGray(p=0.1),
            ]
        )
    else:
        raise NotImplementedError

    for image, label in data_generator:
        if random.random() < p:
            image = transforms(image=image)["image"]
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


def letterbox_collate_fn(batch):
    """
    letterbox collate function for torch.utils.dataloader()
    pad inputs & labels to max-size(32x) of batch
    label: N - <c, x, y, x, y>
    """

    img_batch, _ = zip(*batch)  # transposed
    max_h = math.ceil(max([x.shape[-2] for x in img_batch]) / 32) * 32
    max_w = math.ceil(max([x.shape[-1] for x in img_batch]) / 32) * 32
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


def collate_dataloader(data_generator, batch_size, collate_fn):
    # works exactly as torch.utils.data.Dataloader (without workers & shuffle)
    assert isinstance(batch_size, int)
    assert batch_size > 0
    batch_buffer = []
    for data in data_generator:
        batch_buffer.append(data)
        if len(batch_buffer) >= batch_size:
            yield collate_fn(batch_buffer)
            batch_buffer.clear()
