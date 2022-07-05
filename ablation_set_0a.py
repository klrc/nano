from arch.control_unit.detection_training_control import PresetScratch, PresetFineTuning, TrainingControlUnit
from arch.model.object_detection.yolov5 import yolov5n

if __name__ == "__main__":
    # global settings
    PresetFineTuning.max_epoch = 200

    # pre-training from scratch
    settings = PresetScratch()
    settings.trainset_path = "../datasets/coco/train2017.txt"
    settings.valset_path = "../datasets/coco/val2017.txt"
    settings.save_dir = "runs/yolov5n_scratch"
    model = yolov5n(num_classes=80, weights=".yolov5_checkpoints/yolov5n_sd.pt")
    TrainingControlUnit(model, settings).run(device="0")

    # fine-tuning ablation study: label smoothing
    settings = PresetFineTuning()
    settings.trainset_path = "../datasets/coco/train2017.txt"
    settings.valset_path = "../datasets/coco/val2017.txt"
    settings.save_dir = "runs/yolov5n_abl_smoothing"
    settings.label_smoothing = 0.1
    model = yolov5n(num_classes=80, weights="runs/yolov5n_scratch.0/best.pt")
    TrainingControlUnit(model, settings).run(device="0")

    # fine-tuning ablation study: ExDark dataset
    settings = PresetFineTuning()
    settings.trainset_path = ["../datasets/coco/train2017.txt", "../datasets/ExDark/images/train"]
    settings.valset_path = "../datasets/coco/val2017.txt"
    settings.save_dir = "runs/yolov5n_abl_exdark"
    model = yolov5n(num_classes=80, weights="runs/yolov5n_scratch.0/best.pt")
    TrainingControlUnit(model, settings).run(device="0")

    # fine-tuning ablation study: IndoorCVPR_09 dataset
    settings = PresetFineTuning()
    settings.trainset_path = ["../datasets/coco/train2017.txt", "../datasets/IndoorCVPR_09/images"]
    settings.valset_path = "../datasets/coco/val2017.txt"
    settings.save_dir = "runs/yolov5n_abl_indoorcvpr09"
    model = yolov5n(num_classes=80, weights="runs/yolov5n_scratch.0/best.pt")
    TrainingControlUnit(model, settings).run(device="0")

    # fine-tuning ablation study: fake darkness
    settings = PresetFineTuning()
    settings.trainset_path = "../datasets/coco/train2017.txt"
    settings.valset_path = "../datasets/coco/val2017.txt"
    settings.save_dir = "runs/yolov5n_abl_fake_dark"
    settings.fake_darkness = True
    model = yolov5n(num_classes=80, weights="runs/yolov5n_scratch.0/best.pt")
    TrainingControlUnit(model, settings).run(device="0")

    # fine-tuning ablation study: fake osd
    settings = PresetFineTuning()
    settings.trainset_path = "../datasets/coco/train2017.txt"
    settings.valset_path = "../datasets/coco/val2017.txt"
    settings.save_dir = "runs/yolov5n_abl_fake_osd"
    settings.fake_osd = True
    model = yolov5n(num_classes=80, weights="runs/yolov5n_scratch.0/best.pt")
    TrainingControlUnit(model, settings).run(device="0")
