from object_detection.training_utils import train, DefaultSettings
from object_detection.yolov5_ultralytics import yolov5s, yolov5n
from object_detection.yolov5_ghost import yolov5s_ghost, yolov5n_ghost
from object_detection.yolov5_vovnet import yolov5n_vovnet

if __name__ == "__main__":
    settings = DefaultSettings()
    settings.trainset_path = "../datasets/coco/train2017.txt"
    settings.valset_path = "../datasets/coco/val2017.txt"
    settings.save_dir = "runs/yolov5s"
    settings.patience = 10
    settings.max_epoch = 100
    model = yolov5s(num_classes=80, weights="yolov5s_sd.pt")
    train(model, settings, device="0")

    settings = DefaultSettings()
    settings.trainset_path = "../datasets/coco/train2017.txt"
    settings.valset_path = "../datasets/coco/val2017.txt"
    settings.save_dir = "runs/yolov5n"
    settings.patience = 10
    settings.max_epoch = 100
    model = yolov5n(num_classes=80, weights="yolov5n_sd.pt")
    train(model, settings, device="0")

    settings = DefaultSettings()
    settings.trainset_path = "../datasets/coco/train2017.txt"
    settings.valset_path = "../datasets/coco/val2017.txt"
    settings.save_dir = "runs/yolov5s_ghost"
    settings.patience = 20
    model = yolov5s_ghost(num_classes=80)
    train(model, settings, device="0")

    settings = DefaultSettings()
    settings.trainset_path = "../datasets/coco/train2017.txt"
    settings.valset_path = "../datasets/coco/val2017.txt"
    settings.save_dir = "runs/yolov5n_ghost"
    settings.patience = 20
    model = yolov5n_ghost(num_classes=80)
    train(model, settings, device="0")

    settings = DefaultSettings()
    settings.trainset_path = "../datasets/coco/train2017.txt"
    settings.valset_path = "../datasets/coco/val2017.txt"
    settings.save_dir = "runs/yolov5n_vovnet"
    settings.patience = 20
    model = yolov5n_vovnet(num_classes=80)
    train(model, settings, device="0")
