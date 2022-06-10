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
    from object_detection.training_utils.general import create_dataloader, xywhn2xyxy
    import test_utils.visualize as V
    import cv2

    s = DefaultSettings()
    train_loader, _ = create_dataloader(s.valset_path, training=True, settings=s)
    s.imgsz = 640
    for imgs, targets, _, _ in train_loader:
        imgs = imgs.float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
        targets[:, 2:] = xywhn2xyxy(targets[:, 2:], 640, 640)
        for i, img in enumerate(imgs):
            img = img.unsqueeze(0)
            boxes = targets[targets[:, 0] == i, 1:]
            labels = [s.names[int(x)] for x in boxes[:, 0]]
            canvas = V.Canvas(img[0])
            canvas.draw_boxes_with_label(boxes[:, 1:], labels)
            cv2.imwrite("test.png", canvas.image)
            break
        break
        # print(targets.shape)
        # print(imgs.shape)
        # break
