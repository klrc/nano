from arch.control_unit.detection_training_control import PresetScratch, PresetFineTuning, TrainingControlUnit
from arch.model.object_detection.yolov5 import yolov5n

if __name__ == "__main__":
    # train yolov5n from scratch
    settings = PresetScratch()
    settings.trainset_path = "../datasets/coco/train2017.txt"
    settings.valset_path = "../datasets/coco/val2017.txt"
    settings.save_dir = "runs/yolov5n_scratch"
    model = yolov5n(num_classes=80, weights=".yolov5_checkpoints/yolov5n_sd.pt")
    TrainingControlUnit(model, settings).run(device="0")

    # ablation study: label_smoothing
    settings = PresetFineTuning()
    settings.trainset_path = "../datasets/coco/train2017.txt"
    settings.valset_path = "../datasets/coco/val2017.txt"
    settings.save_dir = "runs/yolov5n_abl_smooth"
    settings.label_smoothing = 0.1
    model = yolov5n(num_classes=80, weights="runs/yolov5n_scratch.0/best.pt")
    TrainingControlUnit(model, settings).run(device="0")

    # ablation study: fake_darkness
    settings = PresetFineTuning()
    settings.trainset_path = "../datasets/coco/train2017.txt"
    settings.valset_path = "../datasets/coco/val2017.txt"
    settings.save_dir = "runs/yolov5n_abl_fake_darkness"
    settings.fake_darkness = True
    model = yolov5n(num_classes=80, weights="runs/yolov5n_scratch.0/best.pt")
    TrainingControlUnit(model, settings).run(device="0")

    # ablation study: fake_osd
    settings = PresetFineTuning()
    settings.trainset_path = "../datasets/coco/train2017.txt"
    settings.valset_path = "../datasets/coco/val2017.txt"
    settings.save_dir = "runs/yolov5n_abl_fake_osd"
    settings.fake_osd = True
    model = yolov5n(num_classes=80, weights="runs/yolov5n_scratch.0/best.pt")
    TrainingControlUnit(model, settings).run(device="0")
