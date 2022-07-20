import glob
import os
import random
from functools import reduce
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm

from .albumentation_presets import Albumentations
from .augment_hsv import AugmentHSV
from .autoanchor import check_anchors
from .infinite_dataloader import InfiniteDataLoader
from .mosaic import Mosaic
from .rand_flip import RandomFlip
from .rand_perspective import RandomPerspective
from .utils import BAR_FORMAT, IMG_FORMATS, NUM_THREADS, get_hash, img2label_paths, letterbox, mixup, verify_image_label, xywhn2xyxy, xyxy2xywhn


class Yolo5Dataset(Dataset):
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version

    def __init__(
        self,
        dataset_path,
        image_size,
        batch_size,
        augment,
        rect,
        mosaic=1.0,  # image mosaic (probability)
        mixup=0.0,  # image mixup (probability)
        degrees=0.0,  # image rotation (+/- deg)
        translate=0.1,  # image translation (+/- fraction)
        scale=0.5,  # image scale (+/- gain)
        shear=0.0,  # image shear (+/- deg)
        perspective=0.0,  # image perspective (+/- fraction), range 0-0.001
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,  # image flip up-down (probability)
        fliplr=0.5,  # image flip left-right (probability)
        fake_darkness=False,
        cache_images=False,
        stride=32,
        pad=0.0,
        prefix="",
    ):
        # Load data into memory
        try:
            f = []  # image files
            for p in dataset_path if isinstance(dataset_path, list) else [dataset_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f"{prefix}{p} does not exist")
            self.im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert self.im_files, f"{prefix}No images found"
        except Exception as e:
            raise Exception(f"{prefix}Error loading data from {dataset_path}: {e}")

        # Check cache
        self.label_files = img2label_paths(self.im_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix(".cache")
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache["version"] == self.cache_version  # same version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # same hash
        except Exception:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupt"
            tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=BAR_FORMAT)  # display cache results
            if cache["msgs"]:
                logger.info("\n".join(cache["msgs"]))  # display warnings
        assert nf > 0 or not augment, f"{prefix}No labels in {cache_path}. Can not train without labels."

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.im_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = segment[j]

        # Rectangular Training
        self.rect = rect
        if rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.im_files = [self.im_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]
            self.batch_shapes = np.ceil(np.array(shapes) * image_size / stride + pad).astype(np.int) * stride

        # Cache images into RAM/disk for faster training (WARNING: large datasets may exceed system resources)
        self.ims = [None] * n
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        if cache_images:
            gb = 0  # Gigabytes of cached images
            self.im_hw0, self.im_hw = [None] * n, [None] * n
            fcn = self.cache_images_to_disk if cache_images == "disk" else self.load_image
            results = ThreadPool(NUM_THREADS).imap(fcn, range(n))
            pbar = tqdm(enumerate(results), total=n, bar_format=BAR_FORMAT)
            for i, x in pbar:
                if cache_images == "disk":
                    gb += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    gb += self.ims[i].nbytes
                pbar.desc = f"{prefix}Caching images ({gb / 1E9:.1f}GB {cache_images})"
            pbar.close()

        # Pipeline definition
        self.image_size = image_size
        self.augment = augment
        if not augment or rect:  # load 4 images at a time into a mosaic (only during training)
            mosaic = 0
        self.mosaic = mosaic
        self.mosaic_pipeline = [
            Mosaic(image_size),
            RandomPerspective(
                degrees=degrees,
                translate=translate,
                scale=scale,
                shear=shear,
                perspective=perspective,
                border=[-image_size // 2, -image_size // 2],
            ),
        ]
        self.mixup = mixup
        self.single_pipeline = [
            RandomPerspective(degrees=degrees, translate=translate, scale=scale, shear=shear, perspective=perspective),
        ]
        self.post_pipeline = [
            Albumentations(fake_darkness=fake_darkness),
            AugmentHSV(hgain=hsv_h, sgain=hsv_s, vgain=hsv_v),
            RandomFlip(ud=flipud, lr=fliplr),
        ]

    def cache_labels(self, path=Path("./labels.cache"), prefix=""):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(
                pool.imap(verify_image_label, zip(self.im_files, self.label_files, repeat(prefix))), desc=desc, total=len(self.im_files), bar_format=BAR_FORMAT
            )
            for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [lb, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupt"

        pbar.close()
        if msgs:
            logger.info("\n".join(msgs))
        if nf == 0:
            logger.warning(f"{prefix}WARNING: No labels found in {path}.")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        x["version"] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
            logger.info(f"{prefix}New cache created: {path}")
        except Exception as e:
            logger.warning(f"{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}")  # not writeable
        return x

    def load_single(self, index):
        # Load image
        image, _, (h, w) = self.load_image(index)
        # Letterbox
        shape = self.batch_shapes[self.batch[index]] if self.rect else self.image_size  # final letterboxed shape
        image, ratio, pad = letterbox(image, shape, auto=False, scaleup=self.augment)
        # shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
        labels = self.labels[index].copy()
        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
        # Single Augmentation Pipeline
        return reduce(lambda x, y: y(*x), self.single_pipeline, (image, labels))

    def load_mosaic(self, index):
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        image_list = [self.load_image(i)[0] for i in indices]
        labels_list = [self.labels[i].copy() for i in indices]
        # Mosaic Augmentation Pipeline
        return reduce(lambda x, y: y(*x), self.mosaic_pipeline, (image_list, labels_list))

    def load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
        im, f, fn = (self.ims[i], self.im_files[i], self.npy_files[i])
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                assert im is not None, f"Image Not Found {f}"
            h0, w0 = im.shape[:2]  # orig hw
            r = self.image_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        else:
            return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def cache_images_to_disk(self, i):
        # Saves an image as an *.npy file for faster loading
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]))

    def __len__(self):
        return len(self.im_files)

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        # Data Loader Pipeline
        if random.random() < self.mosaic:  # Load Mosaic
            image, labels = self.load_mosaic(index)
            if random.random() < self.mixup:  # MixUp augmentation
                image, labels = mixup(image, labels, *self.load_mosaic(random.randint(0, self.n - 1)))
        else:  # Load Single Image
            image, labels = self.load_single(index)

        # Augmentation Pipeline
        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=image.shape[1], h=image.shape[0], clip=True, eps=1e-3)
        if self.augment:
            image, labels = reduce(lambda x, y: y(*x), self.post_pipeline, (image, labels))
        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = np.ascontiguousarray(image)
        return torch.from_numpy(image).float() / 255, labels_out

    @staticmethod
    def collate_fn(batch):
        im, label = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0)


class Yolo5Dataloader:
    def __init__(
        self,
        dataset_path,
        image_size,
        batch_size,
        workers=0,
        mosaic=1.0,  # image mosaic (probability)
        mixup=0.0,  # image mixup (probability)
        degrees=0.0,  # image rotation (+/- deg)
        translate=0.1,  # image translation (+/- fraction)
        scale=0.5,  # image scale (+/- gain)
        shear=0.0,  # image shear (+/- deg)
        perspective=0.0,  # image perspective (+/- fraction), range 0-0.001
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,  # image flip up-down (probability)
        fliplr=0.5,  # image flip left-right (probability)
        fake_darkness=False,
        training=False,
        cache_images=False,
    ):
        if training:
            augment, shuffle, rect, pad = True, True, False, 0.0  # rect is incompatible with DataLoader shuffle
        else:
            augment, shuffle, rect, pad = False, False, True, 0.0
            batch_size *= 2
            workers *= 2
        dataset = Yolo5Dataset(
            dataset_path=dataset_path,
            image_size=image_size,
            batch_size=batch_size,
            augment=augment,
            rect=rect,
            mosaic=mosaic,
            mixup=mixup,
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            perspective=perspective,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            flipud=flipud,
            fliplr=fliplr,
            fake_darkness=fake_darkness,
            cache_images=cache_images,
            pad=pad,
        )
        batch_size = min(batch_size, len(dataset))
        nd = torch.cuda.device_count()  # number of CUDA devices
        nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
        self.dataset = dataset
        self.batch_size = batch_size
        self.image_size = image_size
        self.loader = InfiniteDataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=nw, sampler=None, pin_memory=True, collate_fn=dataset.collate_fn
        )

    def auto_anchor(
        self,
        anchors=(
            (10, 13, 16, 30, 33, 23),
            (30, 61, 62, 45, 59, 119),
            (116, 90, 156, 198, 373, 326),
        ),
        strides=(8, 16, 32),
        thr=4.0,
    ):
        if not isinstance(anchors, torch.Tensor):
            anchors = torch.tensor(anchors).float().view(len(anchors), -1, 2)
        if not isinstance(strides, torch.Tensor):
            strides = torch.tensor(strides)  # strides computed during build
        return check_anchors(self.dataset, anchors.cpu(), strides.cpu(), thr, self.image_size)

    def class_frequency(self, nc):
        # https://arxiv.org/abs/1708.02002 section 3.3
        cf = torch.bincount(torch.tensor(np.concatenate(self.dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.0
        return cf

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for _ in range(len(self.loader)):
            yield next(self.loader.iterator)
