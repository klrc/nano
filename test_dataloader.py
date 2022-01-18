import cv2
import random

from nano.datasets.coco_box2d import MSCOCO
from nano.datasets.coco_box2d_transforms import Affine, Albumentations, HSVTransform, RandomScale, SizeLimit, ToTensor, Mosaic4
from nano.datasets.coco_box2d_visualize import draw_bounding_boxes
from nano.datasets.class_utils import coco_classes, voc_classes, c26_classes, _coco_to_c26, _voc_to_c26, create_class_mapping


# preset configurations
img_root = "../datasets/coco3/images/train"
label_root = "../datasets/coco3/labels/train"
names = ["person", "bike", "car"]


def test_load():
    dataset = MSCOCO(img_root, label_root)
    dataset = ToTensor(dataset)
    rand = random.randint(0, len(dataset) - 1)
    img, labels = dataset.__getitem__(rand)
    str_labels = [names[x] for x in labels[..., 0].cpu().int()]
    cv_img = draw_bounding_boxes(img, boxes=labels[..., 1:], boxes_label=str_labels)
    cv2.imwrite("test_load.png", cv_img)


def test_classmapping():
    img_root_cm = "../datasets/VOC/images/train2012"
    label_root_cm = "../datasets/VOC/labels/train2012"
    names_cm = c26_classes
    class_mapping = create_class_mapping(voc_classes, c26_classes, _voc_to_c26)
    dataset = MSCOCO(img_root_cm, label_root_cm, class_map=class_mapping)
    dataset = SizeLimit(dataset, limit=10, targets=(0, 1, 2))
    dataset = ToTensor(dataset)
    for i in range(10):
        rand = random.randint(0, len(dataset) - 1)
        img, labels = dataset.__getitem__(rand)
        str_labels = [names_cm[x] for x in labels[..., 0].cpu().int()]
        cv_img = draw_bounding_boxes(img, boxes=labels[..., 1:], boxes_label=str_labels)
        cv2.imwrite(f"test_classmapping_{i}.png", cv_img)


def test_hsvtransform():
    dataset = MSCOCO(img_root, label_root)
    dataset = HSVTransform(dataset, p=1)
    dataset = ToTensor(dataset)
    for i in range(10):
        rand = random.randint(0, len(dataset) - 1)
        img, labels = dataset.__getitem__(rand)
        str_labels = [names[x] for x in labels[..., 0].cpu().int()]
        cv_img = draw_bounding_boxes(img, boxes=labels[..., 1:], boxes_label=str_labels)
        cv2.imwrite(f"test_hsvtransform_{i}.png", cv_img)


def test_randomscale():
    dataset = MSCOCO(img_root, label_root)
    dataset = RandomScale(dataset, 0.2, 1, p=1)
    dataset = ToTensor(dataset)
    for i in range(10):
        rand = random.randint(0, len(dataset) - 1)
        img, labels = dataset.__getitem__(rand)
        str_labels = [names[x] for x in labels[..., 0].cpu().int()]
        cv_img = draw_bounding_boxes(img, boxes=labels[..., 1:], boxes_label=str_labels)
        cv2.imwrite(f"test_randomscale_{i}.png", cv_img)


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
    dataset = Affine(dataset, p_flip=0.5, p_shear=1, max_shear=0.5)
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

    class_names = c26_classes
    imgs_root = "../datasets/coco/images/train2017"
    annotations_root = "../datasets/coco/labels/train2017"
    coco_c26_mapping = create_class_mapping(coco_classes, c26_classes, _coco_to_c26)
    coco = MSCOCO(imgs_root=imgs_root, annotations_root=annotations_root, min_size=416, class_map=coco_c26_mapping)
    imgs_root = "../datasets/VOC/images/train2012"
    annotations_root = "../datasets/VOC/labels/train2012"
    voc_c26_mapping = create_class_mapping(voc_classes, c26_classes, _voc_to_c26)
    voc = MSCOCO(imgs_root=imgs_root, annotations_root=annotations_root, min_size=416, class_map=voc_c26_mapping)
    base = coco + voc

    base = SizeLimit(base, 80000, targets=(0, 1, 2))
    base = RandomScale(base, p=1)
    base = Affine(base, p_flip=0.5, p_shear=0.2)
    base = HSVTransform(base, p=0.2)
    base = Albumentations(base, "random_blind")
    base = Mosaic4(base, img_size=448)
    dataset = ToTensor(base)

    for i in range(10):
        rand = random.randint(0, len(dataset) - 1)
        img, labels = dataset.__getitem__(rand)
        str_labels = [class_names[x] for x in labels[..., 0].cpu().int()]
        cv_img = draw_bounding_boxes(img, boxes=labels[..., 1:], boxes_label=str_labels)
        cv2.imwrite(f"test_combination_{i}.png", cv_img)


def summary_bbox_size():
    import tqdm

    dataset = MSCOCO(img_root, label_root, min_size=416)
    dataset = SizeLimit(dataset, limit=20000)
    ratios = {}
    for i in tqdm.tqdm(range(len(dataset))):
        img, labels = dataset.__getitem__(i)
        for c, x1, y1, x2, y2 in labels:
            ratio = (y2 - y1) / (x2 - x1)
            if c not in ratios:
                ratios[c] = []
            ratios[c].append(ratio)

    for c, rarray in ratios.items():
        print(c, sum(rarray) / len(rarray))


if __name__ == "__main__":
    # test_load()
    # test_hsvtransform()
    # test_randomscale()
    # test_classmapping()
    # test_mosaic4()
    # test_affine()
    # test_albumentations()
    # test_sizelimit()
    test_combination()
    # summary_bbox_size()
