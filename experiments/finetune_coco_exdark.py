from object_detection.training_utils import train, DefaultSettings
from object_detection.yolov5_ultralytics import yolov5n, yolov5s

if __name__ == "__main__":
    # fine-tune (more data & strong augmentation)
    DefaultSettings.trainset_path = [
        "../datasets/coco/train2017.txt",
        "../datasets/ExDark/images/train",
    ]
    DefaultSettings.valset_path = ["../datasets/coco/val2017.txt"]
    DefaultSettings.lr0 = 0.003
    DefaultSettings.momentum = 0.75
    DefaultSettings.weight_decay = 0.00025
    DefaultSettings.mosaic = 0.9
    DefaultSettings.lrf = 0.15
    DefaultSettings.scale = 0.75
    DefaultSettings.mixup = 0.04
    DefaultSettings.fake_osd = True
    DefaultSettings.fake_darkness = True

    # start training
    settings = DefaultSettings()
    settings.save_dir = "runs/yolov5n"
    model = yolov5n(num_classes=80, weights="yolov5n_sd.pt")
    train(model, settings, device="0")

    settings = DefaultSettings()
    settings.save_dir = "runs/yolov5s"
    model = yolov5s(num_classes=80, weights="yolov5s_sd.pt")
    train(model, settings, device="0")
