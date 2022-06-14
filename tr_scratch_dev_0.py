from object_detection.training_utils import train, DefaultSettings
from object_detection.yolov5_ultralytics import yolov5s, yolov5n
from object_detection.yolov5_ghost import yolov5s_ghost, yolov5n_ghost
from object_detection.yolov5_vovnet import yolov5n_vovnet

if __name__ == "__main__":
    settings = DefaultSettings()
    settings.trainset_path = "../datasets/coco/train2017.txt"
    settings.valset_path = "../datasets/coco/val2017.txt"
    settings.save_dir = "runs/ultralytics_yolov5s"
    model = yolov5s(num_classes=80, weights="yolov5s_sd.pt")
    train(model, settings, device="0")

    settings = DefaultSettings()
    settings.trainset_path = "../datasets/coco/train2017.txt"
    settings.valset_path = "../datasets/coco/val2017.txt"
    settings.save_dir = "runs/ultralytics_yolov5n"
    model = yolov5n(num_classes=80, weights="yolov5n_sd.pt")
    train(model, settings, device="0")
