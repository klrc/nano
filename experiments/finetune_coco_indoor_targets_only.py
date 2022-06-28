from object_detection.training_utils.default_settings import DefaultSettings
from object_detection.training_utils.general import convert_dataset_labels
from object_detection.training_utils.trainer import train
from object_detection.yolov5_ultralytics import yolov5n, yolov5s


if __name__ == "__main__":
    # process MSCOCO
    mscoco_label_mapper = {
        DefaultSettings.names.index("person"): 0,
        DefaultSettings.names.index("bicycle"): 1,
        DefaultSettings.names.index("car"): 2,
        DefaultSettings.names.index("motorcycle"): 3,
        DefaultSettings.names.index("bus"): 4,
        DefaultSettings.names.index("truck"): 5,
    }
    convert_dataset_labels("../datasets/coco/labels", mscoco_label_mapper)
    convert_dataset_labels("../datasets/IndoorCVPR_09/labels", mscoco_label_mapper, dataset_dirs=("",))
    # global settings: fine-tune (more data & strong augmentation)
    DefaultSettings.names = ["person", "bicycle", "car", "motorcycle", 'bus', 'truck']
    DefaultSettings.nc = 6
    DefaultSettings.trainset_path = [
        "../datasets/coco/train2017.txt",
        "../datasets/IndoorCVPR_09/images",
    ]
    DefaultSettings.valset_path = [
        "../datasets/coco/val2017.txt",
    ]
    DefaultSettings.lr0 = 0.003
    DefaultSettings.momentum = 0.75
    DefaultSettings.weight_decay = 0.00025
    DefaultSettings.lrf = 0.15

    # start training
    settings = DefaultSettings()
    settings.save_dir = "runs/yolov5n"
    model = yolov5n(num_classes=6, weights="runs/yolov5n.1/best.pt")
    train(model, settings, device="0")

    settings = DefaultSettings()
    settings.save_dir = "runs/yolov5s"
    model = yolov5s(num_classes=6, weights="runs/yolov5s.2/best.pt")
    train(model, settings, device="0")
