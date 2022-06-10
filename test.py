from object_detection.training_utils import train, DefaultSettings
from object_detection.yolov5_ultralytics import ultralytics_yolov5s

if __name__ == "__main__":
    # pretraining
    # settings = DefaultSettings()
    # settings.trainset_path = '../datasets/coco/train2017.txt'
    # settings.valset_path = '../datasets/coco/val2017.txt'
    # settings.batch_size = 64
    # settings.input_shape = (1, 3, 384, 640)
    # settings.imgsz = 640
    # settings.save_dir = 'runs/ultralytics_yolov5n'
    # model = ultralytics_yolov5n(num_classes=80, weights='yolov5n.pt')
    # train(model, settings, device="0")

    # fine-tune (more data & strong augmentation)
    settings = DefaultSettings()
    settings.trainset_path = [
        '../datasets/coco/train2017.txt',
        '../datasets/ExDark/images/train',
    ]
    settings.valset_path = [
        '../datasets/coco/val2017.txt'
    ]
    settings.lr0 = 0.003
    settings.scale = 0.9
    settings.copy_paste = 0.1
    settings.fake_osd = True
    settings.fake_darkness = True
    settings.batch_size = 64
    settings.input_shape = (1, 3, 384, 640)
    settings.imgsz = 640
    settings.save_dir = 'runs/ultralytics_yolov5s'
    model = ultralytics_yolov5s(num_classes=80, weights='runs/ultralytics_yolov5s.0/best.pt')
    train(model, settings, device="0")
