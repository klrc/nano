import torch
from object_detection.training_utils import train, DefaultSettings
from object_detection.yolov5_vovnet import VoVYOLO


if __name__ == "__main__":
    settings = DefaultSettings()
    # settings.nc = 6
    # settings.names = ('person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck')
    # settings.trainset_path = '../datasets/coco/train2017.txt'
    # settings.valset_path = '../datasets/coco/val2017.txt'
    settings.batch_size = 4
    settings.input_shape = (1, 3, 384, 640)
    settings.save_dir = 'runs/test'
    settings.conf_thres = 0.1
    # settings.changes = 'vovnet:relu -> relu6'
    model = VoVYOLO(num_classes=80)
    model.load_state_dict(torch.load('runs/test.1/last.pt'))
    train(model, settings, device="cpu")
