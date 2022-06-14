from object_detection.training_utils import train, DefaultSettings
from object_detection.yolov5_ultralytics import yolov5n, yolov5s

if __name__ == "__main__":
    # fine-tune (more data & strong augmentation)
    settings = DefaultSettings()
    settings.trainset_path = [
        "../datasets/coco/train2017.txt",
        "../datasets/ExDark/images/train",
    ]
    settings.valset_path = ["../datasets/coco/val2017.txt"]
    settings.lr0 = 0.003
    settings.momentum = 0.75
    settings.weight_decay = 0.00025
    settings.mosaic = 0.9
    settings.lrf = 0.15
    settings.scale = 0.75
    settings.mixup = 0.04
    settings.fake_osd = True
    settings.fake_darkness = True
    settings.save_dir = "runs/yolov5s"
    model = yolov5s(num_classes=80, weights="runs/yolov5s.0/best.pt")
    train(model, settings, device="0")

    settings = DefaultSettings()
    settings.trainset_path = [
        "../datasets/coco/train2017.txt",
        "../datasets/ExDark/images/train",
    ]
    settings.valset_path = ["../datasets/coco/val2017.txt"]
    settings.lr0 = 0.003
    settings.momentum = 0.75
    settings.weight_decay = 0.00025
    settings.mosaic = 0.9
    settings.lrf = 0.15
    settings.scale = 0.75
    settings.mixup = 0.04
    settings.fake_osd = True
    settings.fake_darkness = True
    settings.save_dir = "runs/yolov5n"
    model = yolov5n(num_classes=80, weights="runs/yolov5n.0/best.pt")
    train(model, settings, device="0")
