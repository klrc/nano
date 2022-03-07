import os
import random
import cv2
import numpy as np
from typing import List

import nano.data.transforms as T
from torch.utils.data import DataLoader, Dataset
import xml.etree.cElementTree as ET


class Seed:
    """
    The primitive seed form for DatasetModule.
    Produces the very original data for object detection,
    In this project, __getitem__() should return (np_image_bgr, np_label_abs_cxyxy)
    """

    def __init__(self) -> None:
        pass

    def __len__(self):
        raise NotImplementedError("Seed function __len__() not implemented")

    def __getitem__(self, index):
        raise NotImplementedError("Seed function __getitem__() not implemented")


class DatasetModule(Dataset):
    """
    The Dataset module for embedded transformations,
    compatible with pytorch dataloader(dataset).
    """

    def __init__(self) -> None:
        self.formula = []

    def add_seed(self, seed, *transforms):
        self.formula.append((seed, transforms))

    def __len__(self):
        return sum([len(x) for x, _ in self.formula])

    def __getitem__(self, index):
        for seed, transforms in self.formula:
            if index >= len(seed):
                index -= len(seed)
                continue
            return self.__feed__(seed, index, transforms)

    def __feed__(self, seed, index, transforms: List[T.TransformFunction]):
        if len(transforms) == 0:
            return seed.__getitem__(index)
        t = transforms[-1]
        if random.random() > t.p:  # skip transform
            return self.__feed__(seed, index, transforms[:-1])
        if t.feed_samples == 1:
            data = self.__feed__(seed, index, transforms[:-1])
        else:  # for transform with multiple feeds
            index_list = [index]
            index_list += [random.randint(0, len(seed) - 1) for _ in range(t.feed_samples - 1)]
            data = [self.__feed__(seed, x, transforms[:-1]) for x in index_list]
        return t(data)

    def __chain__(self, index, hooks: List[T.TransformFunction]):
        if len(hooks) == 0:
            return self.data[index]
        hook_fn = hooks[-1]
        if random.random() <= hook_fn.p:
            if hook_fn.feed_samples > 1:
                index_list = [index]
                index_list += [random.randint(0, len(self) - 1) for _ in range(hook_fn.feed_samples - 1)]
                data = [self.__chain__(x, hooks[:-1]) for x in index_list]
            else:
                data = self.__chain__(index, hooks[:-1])
            return hook_fn(data)
        else:
            return self.__chain__(index, hooks[:-1])

    def as_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)


class MSCOCOSeed(Seed):
    """
    Seed for ultralytics-yolov5 MSCOCO format,
    The directory should contain:
        1. *.png    raw RGB image
        2. *.txt    class_id and bounding boxes in lines (cxywh_relative)
    """

    def __init__(self, *roots) -> None:
        super().__init__()
        self._data = []
        # generate detction dataset items
        unpaired = {}
        for root in roots:
            # extract files from specified root
            if os.path.exists(root):
                for fname in os.listdir(root):
                    path = f"{root}/{fname}"
                    name, _ = fname.split(".")
                    if name in unpaired:
                        self._data.append((unpaired.pop(name), path))
                    else:
                        unpaired[name] = path
        assert len(self._data) > 0, f"No samples found in root {roots}"

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        # load unsorted data (image+label) into numpy
        data = self._data[index]
        image, label = sorted(data, key=lambda x: x.split(".")[-1])
        image = cv2.imread(image)  # BGR
        # process label
        # relative cxywh -> absolute cxyxy
        image_size_factor = image.shape[:2][::-1]  # orig hw
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
        return image, label


class CAVIARSeed(Seed):
    """
    Seed for CAVIAR dataset (https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/) (98782)
    Containing CCTV fisheye perspective video and annotations.
    Dataset directory structure:
        - video_1
            - xxx.jpg
            - xxx.jpg
            ...
            - xxx.jpg
            - xxxxxx.xml
        - video_2
        - video_3

    XML annotation structure:
    <frame number="12">
        <objectlist>
            <object id="0">
                <orientation>133</orientation>
                <box h="18" w="22" xc="77" yc="66"/>
                <appearance>visible</appearance>
                <hypothesislist>
                    <hypothesis evaluation="1.0" id="1" prev="1.0">
                        <movement evaluation="1.0">walking</movement>
                        <role evaluation="1.0">walker</role>
                        <context evaluation="1.0">walking</context>
                        <situation evaluation="1.0">moving</situation>
                    </hypothesis>
                 </hypothesislist>
            </object>
    * object contains person only
    """

    def __init__(self, root, pick_rate=1) -> None:
        super().__init__()
        self._data = []
        for video_set in os.listdir(root):
            if not os.path.isdir(f"{root}/{video_set}"):
                continue
            files = [f for f in os.listdir(f"{root}/{video_set}")]
            frames = sorted([x for x in files if x.endswith(".jpg")])
            xml_file = [x for x in files if x.endswith(".xml")][0]
            # parse xml
            tree = ET.parse(f"{root}/{video_set}/{xml_file}")
            for i, frame in enumerate(tree.getroot().findall("frame")):
                if i % int(1 / pick_rate) == 0:
                    n = int(frame.attrib["number"])
                    object_list = []
                    for object in frame.find("objectlist").findall("object"):
                        # obj_id = object.attrib["id"]
                        box = object.find("box")
                        h, w, xc, yc = [int(box.attrib[x]) for x in ("h", "w", "xc", "yc")]
                        labels = (0, xc - w * 0.5, yc - h * 0.5, xc + w * 0.5, yc + h * 0.5)
                        object_list.append(labels)
                    self._data.append((f"{root}/{video_set}/{frames[n]}", object_list))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        # load unsorted data (image+label) into numpy
        image, label = self._data[index]
        image = cv2.imread(image)  # BGR
        # process label
        label = np.array(label)
        return image, label


class PETS09Seed(Seed):
    """
    Seed for PETS09-S2 dataset(http://www.milanton.de/data/) (1471)
    Containing CCTV frames and annotations for person tracking.
    Dataset directory structure:
    - S2
        - L1
            - TIME_XX-XX
                - View_XXX
                    - xxx.jpg
                    - xxx.jpg
                    ...
        - L2
        - L3
        - PETS2009-S2L1.xml
        - PETS2009-S2L2.xml
        - PETS2009-S2L3.xml

    * All annotations were done in View001

    XML annotation structure:
    <frame number="0">
        <objectlist>
            <object id="9">
                <box h="75.17" w="31.03" xc="514.7109" yc="195.2731"/>
            </object>
            <object id="15">
                <box h="88.7021" w="32.9129" xc="274.4912" yc="262.9999"/>
            </object>
            <object id="19">
                <box h="81.0743" w="42.3382" xc="654.358" yc="282.4698"/>
            </object>
        </objectlist>
    </frame>

    """

    def __init__(self, root, pick_rate=1) -> None:
        super().__init__()
        self._data = []
        for level in ("L1", "L2", "L3"):
            frames = []
            level_root = f"{root}/S2/{level}"
            for clip in os.listdir(level_root):
                clip_root = f"{root}/S2/{level}/{clip}/View_001"
                if not os.path.isdir(clip_root):
                    continue
                for image in os.listdir(clip_root):
                    if image.endswith("jpg"):
                        frames.append(f"{clip_root}/{image}")
            frames = sorted(frames)
            tree = ET.parse(f"{root}/S2/PETS2009-S2{level}.xml")
            for i, frame in enumerate(tree.getroot().findall("frame")):
                if i % int(1 / pick_rate) == 0:
                    n = int(frame.attrib["number"])
                    object_list = []
                    for object in frame.find("objectlist").findall("object"):
                        # obj_id = object.attrib["id"]
                        box = object.find("box")
                        h, w, xc, yc = [int(float(box.attrib[x])) for x in ("h", "w", "xc", "yc")]
                        labels = (0, xc - w * 0.5, yc - h * 0.5, xc + w * 0.5, yc + h * 0.5)
                        object_list.append(labels)
                    self._data.append((frames[n], object_list))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        # load unsorted data (image+label) into numpy
        image, label = self._data[index]
        image = cv2.imread(image)  # BGR
        # process label
        label = np.array(label)
        return image, label


class VIRATSeed(Seed):
    """
    Seed for VIRAT dataset(https://gitlab.kitware.com/viratdata/viratannotations) (4793)
    Containing high-resolution suvilliance mp4 video

    Dataset structure:
    - aerial
    - ground
        - annotations
            - VIRAT_S_xxxxxxx.viratdata.objects.txt
            - VIRAT_S_xxxxxxx.viratdata.objects.txt
            ...
        - videos_original
            - xxx.mp4
            - xxx.mp4
            ...

    Object file format:
        Files are named as '%s.viratdata.objects.txt'
        Each line captures informabiont about a bounding box of an object (person/car etc) at the corresponding frame.
        Each object track is assigned a unique 'object id' identifier.
        Note that:
        - an object may be moving or static (e.g., parked car).
        - an object track may be fragmented into multiple tracks.

        Object File Columns
        1: Object id        (a unique identifier of an object track. Unique within a file.)
        2: Object duration  (duration of the object track)
        3: Currnet frame    (corresponding frame number)
        4: bbox lefttop x   (horizontal x coordinate of the left top of bbox, origin is lefttop of the frame)
        5: bbox lefttop y   (vertical y coordinate of the left top of bbox, origin is lefttop of the frame)
        6: bbox width       (horizontal width of the bbox)
        7: bbox height      (vertical height of the bbox)
        8: Objct Type       (object type)

        Object Type ID (for column 8 above for object files)
        1: person
        2: car              (usually passenger vehicles such as sedan, truck)
        3: vehicles         (vehicles other than usual passenger cars. Examples include construction vehicles)
        4: object           (neither car or person, usually carried objects)
        5: bike, bicylces   (may include engine-powered auto-bikes)

    """

    def __init__(self, root, pick_rate=1) -> None:
        super().__init__()
        self._data = []
        video_root = f"{root}/ground/videos_original"
        xml_root = f"{root}/ground/annotations"
        # os.system(f"rm {video_root}/._*.mp4")
        for clip in os.listdir(video_root):
            # ----------------------------------------- to generate jpgs from video
            # if clip.endswith('.mp4'):
            #     print(f'processing {clip}')
            #     os.system(f"mkdir {video_root}/{clip.split('.')[0]}")
            #     vidcap = cv2.VideoCapture(f"{video_root}/{clip}")
            #     success, image = vidcap.read()
            #     count = 0
            #     while success:
            #         if count % 10 == 0:
            #             cv2.imwrite(f"{video_root}/{clip.split('.')[0]}/{count}.jpg", image)  # save frame as JPEG file
            #         success, image = vidcap.read()
            #         count += 1
            # -------------------------------------------------------------------
            if os.path.isdir(f"{video_root}/{clip}"):
                tree = ET.parse(f"{xml_root}/{clip}.viratdata.objects.xml")
                for i, frame in enumerate(tree.getroot().findall("frame")):
                    if i % int(1 / pick_rate) == 0:
                        n = int(frame.attrib["number"])
                        object_list = []
                        for object in frame.find("objectlist").findall("object"):
                            box = object.find("box")
                            c, x1, y1, x2, y2 = [float(box.attrib[x]) for x in ("c", "x1", "y1", "x2", "y2")]
                            object_list.append((c, x1, y1, x2, y2))
                        self._data.append((f"{video_root}/{clip}/{n}.jpg", object_list))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        # load unsorted data (image+label) into numpy
        image, label = self._data[index]
        image = cv2.imread(image)  # BGR
        # process label
        label = np.array(label)
        return image, label


class EmptyRandChoice(Seed):
    """
    Seed for images with absolutely no objects, usually for negative samples.
    The directory should contain:
        - *.png    raw images
    """

    def __init__(self, *roots, pick_rate=1) -> None:
        super().__init__()
        self._data = []
        frames = []
        for root in roots:
            # extract files from specified root
            if os.path.exists(root):
                os.system(f"rm {root}/._*.jpg")
                for fname in os.listdir(root):
                    if fname.endswith(".jpg"):
                        path = f"{root}/{fname}"
                        frames.append(path)
        if pick_rate < 1:
            frames = random.choices(frames, k=int(len(frames) * pick_rate))
        self._data = frames
        assert len(self._data) > 0, f"No samples found in {roots}"

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        # load unsorted data (image+label) into numpy
        image = cv2.imread(self._data[index])  # BGR
        # process empty label
        label = np.array([])
        return image, label


class IndoorSeed(EmptyRandChoice):
    """
    Seed for Indoor Object Detection dataset(https://zenodo.org/record/2654485#.YgzWw9--tpR) (2213)
    Containing frames for indoor facilities with ACTUALLY NO person in it.
    """

    def __init__(self, root, pick_rate=1) -> None:
        super().__init__(*[f"{root}/sequence_{i}" for i in range(1, 7)], pick_rate=pick_rate)


class SKU110KSeed(EmptyRandChoice):
    """
    Seed for SKU110K dataset(https://github.com/eg4000/SKU110K_CVPR19) (11743)
    Containing supermarket detection data with NEARLY no person in it.
    """

    def __init__(self, root, pick_rate=1) -> None:
        super().__init__(f"{root}/images", pick_rate=pick_rate)


def voc_quick_test_preset(
    target_resolution=(224, 416),
    target_classes="person|bike|car|OOD",
    dataset_root="/your/dataset/root",
):
    from .dataset_info import ClassHub

    factory = DatasetModule()
    factory.add_seed(
        MSCOCOSeed(
            f"{dataset_root}/VOC/images/train2012",
            f"{dataset_root}/VOC/labels/train2012",
        ),
        T.IndexMapping(ClassHub("voc").to(target_classes)),
        T.HorizontalFlip(p=0.5),
        T.Resize(max_size=int(max(target_resolution) * 0.75)),  # 0.75x
        T.RandomScale(min_scale=0.6, max_scale=1),  # 0.8x
        T.RandomAffine(min_scale=0.875, max_scale=1.125, p=0.5),  # 1x
        T.HSVTransform(),
        T.AlbumentationsPreset(),
        T.Mosaic4(mosaic_size=int(max(target_resolution)), min_iou=0.45, p=1),
        T.ToTensor(),
    )
    return factory


def preson_vehicle_detection_preset_mscoco_test(
    target_resolution=(224, 416),
    target_classes="person|bike|car|OOD",
    dataset_root="/your/dataset/root",
):
    from .dataset_info import ClassHub
    factory = DatasetModule()
    factory.add_seed(
        MSCOCOSeed(
            f"{dataset_root}/MSCOCO/val2017",
            f"{dataset_root}/MSCOCO/labels/val2017",
        ),
        T.IndexMapping(ClassHub("coco").to(target_classes)),
        T.Resize(max_size=int(max(target_resolution))),
        T.ToTensor(),
    )
    return factory


def person_vehicle_detection_preset(
    target_resolution=(224, 416),
    target_classes="person|bike|car|OOD",
    dataset_root="/your/dataset/root",
):
    from .dataset_info import ClassHub

    factory = DatasetModule()
    factory.add_seed(
        MSCOCOSeed(
            f"{dataset_root}/VOC/images/train2012",
            f"{dataset_root}/VOC/labels/train2012",
        ),
        T.IndexMapping(ClassHub("voc").to(target_classes)),
        T.HorizontalFlip(p=0.5),
        T.Resize(max_size=int(max(target_resolution) * 0.75)),  # 0.75x
        T.RandomScale(min_scale=0.6, max_scale=1),  # 0.8x
        T.RandomAffine(min_scale=0.9, max_scale=1.1, p=0.1),  # 1x
        T.HSVTransform(p=0.1),
        T.AlbumentationsPreset(),
        T.Mosaic4(mosaic_size=int(max(target_resolution)), min_iou=0.45, p=1),
        T.ToTensor(),
    )
    factory.add_seed(
        MSCOCOSeed(
            f"{dataset_root}/MSCOCO/train2017",
            f"{dataset_root}/MSCOCO/labels/train2017",
        ),
        T.IndexMapping(ClassHub("coco").to(target_classes)),
        T.HorizontalFlip(p=0.5),
        T.Resize(max_size=int(max(target_resolution))),  # 1x
        T.RandomScale(min_scale=0.875, max_scale=1.125),  # 1x
        T.RandomAffine(min_scale=0.9, max_scale=1.1, p=0.1),  # 1x
        T.HSVTransform(p=0.1),
        T.AlbumentationsPreset(),
        T.Mosaic4(mosaic_size=int(max(target_resolution)), min_iou=0.45, p=1),
        T.ToTensor(),
    )
    factory.add_seed(
        CAVIARSeed(f"{dataset_root}/CAVIAR", pick_rate=0.002),
        T.HorizontalFlip(p=0.5),
        T.Resize(max_size=int(max(target_resolution))),  # 1x
        T.RandomAffine(min_scale=0.9, max_scale=1.1, p=0.1),  # 1x
        T.HSVTransform(p=0.1),
        T.AlbumentationsPreset(),
        T.Mosaic4(mosaic_size=int(max(target_resolution)), min_iou=0.45, p=1),
        T.ToTensor(),
    )
    factory.add_seed(
        VIRATSeed(f"{dataset_root}/VIRAT", pick_rate=0.002),
        T.IndexMapping(ClassHub("virat").to(target_classes)),
        T.HorizontalFlip(p=0.5),
        T.Resize(max_size=int(max(target_resolution))),
        T.RandomScale(min_scale=1, max_scale=1.5),
        T.RandomAffine(min_scale=0.9, max_scale=1.1, p=0.5),  # 1x
        T.HSVTransform(p=0.1),
        T.AlbumentationsPreset(),
        T.Mosaic4(mosaic_size=int(max(target_resolution)), min_iou=0.45, p=1),
        T.ToTensor(),
    )
    factory.add_seed(
        PETS09Seed(f"{dataset_root}/Crowd_PETS09", pick_rate=0.005),
        T.HorizontalFlip(p=0.5),
        T.Resize(max_size=int(max(target_resolution))),  # 1x
        T.RandomAffine(min_scale=0.9, max_scale=1.1, p=0.5),  # 1x
        T.HSVTransform(p=0.1),
        T.AlbumentationsPreset(),
        T.Mosaic4(mosaic_size=int(max(target_resolution)), min_iou=0.45, p=1),
        T.ToTensor(),
    )
    factory.add_seed(
        IndoorSeed(f"{dataset_root}/IndoorOD", pick_rate=0.01),
        T.HorizontalFlip(p=0.5),
        T.Resize(max_size=int(max(target_resolution))),  # 1x
        T.RandomAffine(min_scale=0.9, max_scale=1.1, p=0.5),  # 1x
        T.HSVTransform(p=0.1),
        T.AlbumentationsPreset(),
        T.Mosaic4(mosaic_size=int(max(target_resolution)), min_iou=0.45, p=1),
        T.ToTensor(),
    )
    return factory
