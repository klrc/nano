import os
from pycocotools.coco import COCO
from tqdm import tqdm
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import random

sns.set()


class Bbox:
    def __init__(self, name, x, y, w, h):
        self.name = name
        self.x = x
        self.y = y
        self.w = w
        self.h = h


class DataProto:
    def __init__(self, image_path, annotations, dataset) -> None:
        self.image_path = image_path
        self.annotations = annotations
        self.dataset = dataset

    def __str__(self) -> str:
        repr = ""
        for k, v in self.__dict__.items():
            if k == "annotations":
                repr += f"{k}: {str([x.name for x in v])}\n"
            else:
                repr += f"{k}: {v}\n"
        return repr


class DatasetContainer:
    def __init__(self) -> None:
        self.data = []

    def read_meta(self, coco, iid, coco_names):
        img_meta = coco.loadImgs(iid)[0]
        filename = img_meta["file_name"]
        width = img_meta["width"]
        height = img_meta["height"]
        annotations = []
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=iid, iscrowd=None)):
            bbox = ann["bbox"]
            cid = ann["category_id"]
            name = coco_names[cid]
            x = (bbox[0] + bbox[2] / 2.0 - 1) / width
            y = (bbox[1] + bbox[3] / 2.0 - 1) / height
            w = bbox[2] / width
            h = bbox[3] / height
            annotations.append(Bbox(name, x, y, w, h))
        return filename, annotations

    def _coco_loader(self, root, dataset, target_dataset):
        coco = COCO(f"{root}/annotations/instances_{dataset}.json")
        categories = coco.dataset["categories"]
        coco_names = {x["id"]: x["name"] for x in categories}

        iids = []
        for cid in [x["id"] for x in categories]:
            for iid in tqdm(coco.getImgIds(catIds=[cid])):
                iids.append(iid)

        for iid in tqdm(set(iids)):
            filename, annotations = self.read_meta(coco, iid, coco_names)
            image_path = f"{root}/images/{dataset}/{filename}"
            D = DataProto(image_path, annotations, target_dataset)
            self.data.append(D)

    def _voc_loader(self, root, dataset, target_dataset):
        voc_names = [
            "airplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "dining table",
            "dog",
            "horse",
            "motorcycle",
            "person",
            "potted plant",
            "sheep",
            "couch",
            "train",
            "tv",
        ]
        voc_names = {i: x for i, x in enumerate(voc_names)}
        for filename in tqdm(os.listdir(f"{root}/images/{dataset}")):
            image_path = f"{root}/images/{dataset}/{filename}"
            annotations = []
            with open(f"{root}/labels/{dataset}/{filename.split('.')[0]}.txt", "r") as f:
                for line in f.readlines():
                    cid, x, y, w, h = line.split(" ")
                    annotations.append(Bbox(voc_names[int(cid)], x, y, w, h))
            D = DataProto(image_path, annotations, target_dataset)
            self.data.append(D)

    def load_dataset(self, root, dataset, target_dataset):
        if root.endswith("coco"):
            self._coco_loader(root, dataset, target_dataset)
        elif root.endswith("VOC"):
            self._voc_loader(root, dataset, target_dataset)
        print("len(data):", len(self.data))
        print(self.data[-1])

    def export(self, root):
        # clean target root
        if os.path.exists(root):
            shutil.rmtree(root)
        # load true cid mapping
        custom_cids = {
            "person": 0,
            "bicycle": 1,
            "motorcycle": 1,
            "car": 2,
            "bus": 2,
            "truck": 2,
        }
        # build new dataset
        os.makedirs(f"{root}/images/train")
        os.makedirs(f"{root}/labels/train")
        os.makedirs(f"{root}/images/val")
        os.makedirs(f"{root}/labels/val")
        for D in tqdm(self.data):
            D: DataProto
            dataset = D.dataset
            image_path = D.image_path
            image_filename = image_path.split("/")[-1]
            label_filename = image_filename.split(".")[0] + ".txt"
            shutil.copy(image_path, f"{root}/images/{dataset}/{image_filename}")
            with open(f"{root}/{dataset}.txt", "a") as f:
                f.write(f"./images/{dataset}/{image_filename}\n")
            with open(f"{root}/labels/{dataset}/{label_filename}", "w") as f:
                for bbox in D.annotations:
                    bbox: Bbox
                    if bbox.name not in custom_cids:
                        continue
                    line = [custom_cids[bbox.name], bbox.x, bbox.y, bbox.w, bbox.h]
                    f.write(" ".join([str(x) for x in line]) + "\n")
        # finish exporting, return with root path
        return root

    def reduce_instances(self, cut_val=True):
        reduced = []
        for D in self.data:
            if not cut_val and D.dataset == "val":  # skip val dataset
                reduced.append(D)
                continue
            instances_person = 0
            instances_vehicle = 0
            instances_all = 0
            reduced_annotations = []
            for b in D.annotations:
                if float(b.x) * 416 <= 3 or float(b.y) * 416 <= 3:  # reduce extremely small objects
                    continue
                if b.name == "person":
                    instances_person += 1
                if b.name in ["bicycle", "motorcycle", "car", "bus", "truck"]:
                    instances_vehicle += 1
                instances_all += 1
                reduced_annotations.append(b)
            D.annotations = reduced_annotations
            if instances_vehicle == 0:  # reduce non-vehicle images
                if instances_person == 0 and random.random() < 0.5:  # keep 1/2 negativa samples
                    pass
                else:
                    continue
                # continue
            reduced.append(D)
        print("len(reduced):", len(reduced))
        self.data = reduced

    def show_class_histplot(self):
        custom_cids = {
            "person": 0,
            "bicycle": 1,
            "motorcycle": 1,
            "car": 2,
            "bus": 2,
            "truck": 2,
            "other": 3,
        }
        plot_data = []
        for D in self.data:
            for b in D.annotations:
                if b.name not in custom_cids:
                    plot_data.append("other")
                else:
                    plot_data.append(b.name)
        plot_data = [custom_cids[x] for x in plot_data]
        sns.histplot(plot_data, kde=True)
        plt.savefig("runs/misc/class_histplot.png")


c = DatasetContainer()
c.load_dataset("/home/sh/Datasets/VOC", "train2012", "train")
c.load_dataset("/home/sh/Datasets/VOC", "test2007", "train")
c.load_dataset("/home/sh/Datasets/VOC", "val2012", "val")
c.load_dataset("/home/sh/Datasets/coco", "train2017", "train")
c.load_dataset("/home/sh/Datasets/coco", "val2017", "train")

# c.reduce_instances(cut_val=False)
c.show_class_histplot()

c.export("/home/sh/Datasets/coc-f")
