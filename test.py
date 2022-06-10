from object_detection.training_utils import train, DefaultSettings
from object_detection.yolov5_ultralytics import ultralytics_yolov5n

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
    # settings = DefaultSettings()
    # settings.trainset_path = [
    #     '../datasets/coco/train2017.txt',
    #     '../datasets/ExDark/images/train',
    # ]
    # settings.valset_path = [
    #     '../datasets/coco/val2017.txt'
    # ]
    # settings.scale = 0.9
    # settings.copy_paste = 0.1
    # settings.fake_osd = True
    # settings.fake_darkness = True
    # settings.batch_size = 64
    # settings.input_shape = (1, 3, 384, 640)
    # settings.imgsz = 640
    # settings.save_dir = 'runs/ultralytics_yolov5n'
    # model = ultralytics_yolov5n(num_classes=80, weights='yolov5n.pt')
    # train(model, settings, device="0")

    # visualize data
    import random
    random.seed(7)

    from object_detection.training_utils.general import create_dataloader, xywhn2xyxy
    import test_utils.visualize as V
    import cv2

    s = DefaultSettings()
    s.fake_osd = True
    _, dataset = create_dataloader(s.valset_path, training=True, settings=s)
    s.imgsz = 640

    img, target, _, _ = dataset.__getitem__(7)
    img = img.unsqueeze(0).float() / 255
    boxes = xywhn2xyxy(target[:, 2:], 640, 640)
    labels = [s.names[int(x)] for x in target[:, 1]]
    canvas = V.Canvas(img[0])
    canvas.draw_boxes_with_label(boxes, labels)
    cv2.imwrite("fake_osd_on.png", canvas.image)
