from arch.model import yolov5n_4c
from arch.controller.detection_training_4c import DetectionTraining4C
from arch.data.detection_annotation import Yolo5Dataloader
import torch

if __name__ == "__main__":
    model = yolov5n_4c(weight=".yolov5_checkpoints/yolov5n_sd.pt", pretrained=True).replace_activation(torch.nn.SiLU)
    train_loader = Yolo5Dataloader('../datasets/coco/images/train2017', 640, 64)
    val_loader = Yolo5Dataloader('../datasets/coco/images/val2017', 640, 64)

    DT = DetectionTraining4C()
    DT.save_dir = 'runs/yolov5n_4c'
    DT.device = 0
    DT.train(model, train_loader, val_loader, model.head.loss)
