def test():
    from nano.datasets.coco_box2d import MSCOCO
    img_root = '../datasets/coco3/images/train'
    label_root = '../datasets/coco3/labels/train'
    dataset = MSCOCO(img_root, label_root)
    for x in dataset:
        print(x)

test()