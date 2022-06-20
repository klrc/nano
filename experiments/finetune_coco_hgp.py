from object_detection.training_utils.default_settings import DefaultSettings
from object_detection.training_utils.general import convert_dataset_labels
from object_detection.training_utils.trainer import train
from object_detection.yolov5_ultralytics import yolov5n, yolov5s

if __name__ == "__main__":
    # process dataset
    label_mapper = {
        0: DefaultSettings.names.index("cell phone"),  # phone
        1: 80,  # gun
        2: 81,  # hand
    }
    convert_dataset_labels("../datasets/HGP/labels", label_mapper)

    # global settings: fine-tune (more data & strong augmentation)
    DefaultSettings.names.append("gun")
    DefaultSettings.names.append("hand")
    DefaultSettings.nc = 82
    DefaultSettings.trainset_path = [
        "../datasets/coco/train2017.txt",
        "../datasets/HGP/images/train2017",
    ]
    DefaultSettings.valset_path = [
        "../datasets/coco/val2017.txt",
        "../datasets/HGP/images/val2017",
    ]
    DefaultSettings.lr0 = 0.003
    DefaultSettings.momentum = 0.75
    DefaultSettings.weight_decay = 0.00025
    DefaultSettings.lrf = 0.15

    # start training
    settings = DefaultSettings()
    settings.save_dir = "runs/yolov5n"
    model = yolov5n(num_classes=82, weights="runs/yolov5n.1/best.pt")
    train(model, settings, device="0")

    settings = DefaultSettings()
    settings.save_dir = "runs/yolov5s"
    model = yolov5s(num_classes=82, weights="runs/yolov5s.2/best.pt")
    train(model, settings, device="0")
