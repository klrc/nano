from object_detection.training_utils import train, DefaultSettings
from object_detection.yolov5_ultralytics import ultralytics_yolov5n, ultralytics_yolov5s
from object_detection.yolov5_ghost import yolov5_ghost
from object_detection.yolov5_vovnet import yolov5_vovnet

if __name__ == "__main__":
    settings = DefaultSettings()
    settings.trainset_path = '../datasets/coco/train2017.txt'
    settings.valset_path = '../datasets/coco/val2017.txt'
    settings.batch_size = 64
    settings.input_shape = (1, 3, 384, 640)
    settings.imgsz = 640
    settings.save_dir = 'runs/ultralytics_yolov5n'
    model = ultralytics_yolov5n(num_classes=80, weights='yolov5n.pt')
    train(model, settings, device="0")

    settings = DefaultSettings()
    settings.trainset_path = '../datasets/coco/train2017.txt'
    settings.valset_path = '../datasets/coco/val2017.txt'
    settings.batch_size = 64
    settings.input_shape = (1, 3, 384, 640)
    settings.imgsz = 640
    settings.save_dir = 'runs/ultralytics_yolov5s'
    model = ultralytics_yolov5s(num_classes=80, weights='yolov5s.pt')
    train(model, settings, device="0")

    settings = DefaultSettings()
    settings.trainset_path = '../datasets/coco/train2017.txt'
    settings.valset_path = '../datasets/coco/val2017.txt'
    settings.batch_size = 64
    settings.input_shape = (1, 3, 384, 640)
    settings.imgsz = 640
    settings.save_dir = 'runs/yolov5_ghost'
    model = yolov5_ghost(num_classes=80)
    train(model, settings, device="0")

    settings = DefaultSettings()
    settings.trainset_path = '../datasets/coco/train2017.txt'
    settings.valset_path = '../datasets/coco/val2017.txt'
    settings.batch_size = 64
    settings.input_shape = (1, 3, 384, 640)
    settings.imgsz = 640
    settings.save_dir = 'runs/yolov5_vovnet'
    model = yolov5_vovnet(num_classes=80)
    train(model, settings, device="0")
