from arch.control_unit.detection_training_control import PresetScratch, TrainingControlUnit
from arch.model.object_detection.yolov5 import yolov5s

if __name__ == "__main__":
    settings = PresetScratch()
    settings.save_dir = "runs/yolov5s"
    settings.batch_size = 16
    model = yolov5s(num_classes=80)
    tcu = TrainingControlUnit(model, settings)
    tcu.FAST_DEBUG_MODE = True
    tcu.run(device="cpu")
