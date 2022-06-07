from object_detection.training_utils import train, DefaultSettings
from object_detection.yolov5_vovnet import VoVYOLO


if __name__ == "__main__":
    settings = DefaultSettings()
    settings.nc = 6
    settings.names = ('person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck')
    settings.trainset_path = '../datasets/coco/train2017.txt'
    settings.valset_path = '../datasets/coco/val2017.txt'
    settings.batch_size = 64
    settings.input_shape = (1, 3, 384, 640)
    model = VoVYOLO(num_classes=6)
    train(model, settings, device="0")
