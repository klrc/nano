from object_detection.training_utils import train, DefaultSettings
from object_detection.yolov5_ultralytics import yolov5s, yolov5n
from object_detection.yolov5_ghost import yolov5s_ghost, yolov5n_ghost
from object_detection.yolov5_vovnet import yolov5n_vovnet

if __name__ == "__main__":
    settings = DefaultSettings()
    settings.trainset_path = "../datasets/coco/train2017.txt"
    settings.valset_path = "../datasets/coco/val2017.txt"
    settings.save_dir = "runs/ultralytics_yolov5s"
    model = yolov5s_ghost(num_classes=80)
    train(model, settings, device="1")

    settings = DefaultSettings()
    settings.trainset_path = "../datasets/coco/train2017.txt"
    settings.valset_path = "../datasets/coco/val2017.txt"
    settings.save_dir = "runs/yolov5s_ghost"
    model = yolov5n_ghost(num_classes=80)
    train(model, settings, device="1")

    settings = DefaultSettings()
    settings.trainset_path = "../datasets/coco/train2017.txt"
    settings.valset_path = "../datasets/coco/val2017.txt"
    settings.save_dir = "runs/yolov5_vovnet"
    model = yolov5n_vovnet(num_classes=80)
    train(model, settings, device="1")
