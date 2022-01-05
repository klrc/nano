import cv2
import random


from nano.datasets.coco_box2d import MSCOCO
from nano.datasets.coco_box2d_transforms import Affine, Albumentations, SizeLimit, ToTensor, Mosaic4
from nano.datasets.coco_box2d_visualize import draw_bounding_boxes


# preset configurations
img_root = "/home/sh/Datasets/coco3/images/val"
label_root = "/home/sh/Datasets/coco3/labels/val"
names = ["person", "bike", "car"]


def test_load():
    dataset = MSCOCO(img_root, label_root)
    dataset = ToTensor(dataset)
    rand = random.randint(0, len(dataset) - 1)
    img, labels = dataset.__getitem__(rand)
    str_labels = [names[x] for x in labels[..., 0].cpu().int()]
    cv_img = draw_bounding_boxes(img, boxes=labels[..., 1:], boxes_label=str_labels)
    cv2.imwrite("test_load.png", cv_img)


def test_mosaic4():
    dataset = MSCOCO(img_root, label_root, 416)
    dataset = Mosaic4(dataset, 448)
    dataset = ToTensor(dataset)
    for i in range(10):
        rand = random.randint(0, len(dataset) - 1)
        img, labels = dataset.__getitem__(rand)
        str_labels = [names[x] for x in labels[..., 0].cpu().int()]
        cv_img = draw_bounding_boxes(img, boxes=labels[..., 1:], boxes_label=str_labels)
        cv2.imwrite(f"test_mosaic4_{i}.png", cv_img)


def test_affine():
    dataset = MSCOCO(img_root, label_root)
    dataset = Affine(dataset, perspective=1)
    dataset = ToTensor(dataset)
    for i in range(4):
        rand = random.randint(0, len(dataset) - 1)
        img, labels = dataset.__getitem__(rand)
        str_labels = [names[x] for x in labels[..., 0].cpu().int()]
        cv_img = draw_bounding_boxes(img, boxes=labels[..., 1:], boxes_label=str_labels)
        cv2.imwrite(f"test_affine_{i}.png", cv_img)


def test_albumentations():
    dataset = MSCOCO(img_root, label_root)
    dataset = Albumentations(dataset)
    dataset = ToTensor(dataset)
    for i in range(10):
        rand = random.randint(0, len(dataset) - 1)
        img, labels = dataset.__getitem__(rand)
        str_labels = [names[x] for x in labels[..., 0].cpu().int()]
        cv_img = draw_bounding_boxes(img, boxes=labels[..., 1:], boxes_label=str_labels)
        cv2.imwrite(f"test_albumentations_{i}.png", cv_img)


def test_sizelimit():
    dataset = MSCOCO(img_root, label_root)
    dataset = SizeLimit(dataset, limit=5)
    dataset = ToTensor(dataset)
    for i, (img, labels) in enumerate(dataset):
        str_labels = [names[x] for x in labels[..., 0].cpu().int()]
        cv_img = draw_bounding_boxes(img, boxes=labels[..., 1:], boxes_label=str_labels)
        cv2.imwrite(f"test_sizelimit_{i}.png", cv_img)


def test_combination():
    dataset = MSCOCO(img_root, label_root)
    dataset = SizeLimit(dataset, limit=5)
    dataset = Affine(dataset, perspective=1)
    dataset = Albumentations(dataset)
    dataset = Mosaic4(dataset, 448)
    dataset = ToTensor(dataset)
    for i in range(4):
        rand = random.randint(0, len(dataset) - 1)
        img, labels = dataset.__getitem__(rand)
        str_labels = [names[x] for x in labels[..., 0].cpu().int()]
        cv_img = draw_bounding_boxes(img, boxes=labels[..., 1:], boxes_label=str_labels)
        cv2.imwrite(f"test_combination_{i}.png", cv_img)


if __name__ == "__main__":
    test_load()
    test_mosaic4()
    test_affine()
    test_albumentations()
    test_sizelimit()
    test_combination()
