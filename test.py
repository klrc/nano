# from arch.controller.detection_pipeline import DetectPipeline4C
# from arch.data.video import LiveCamera

from arch.model import yolov5n
from arch.controller.detection_training_41c import DetectionTraining41C
from arch.data.detection_annotation import Yolo5Dataloader
import torch

if __name__ == "__main__":
    model = yolov5n(weight=".yolov5_checkpoints/yolov5n_sd.pt", pretrained=True).replace_activation(torch.nn.SiLU)
    train_loader = Yolo5Dataloader('../datasets/coco128/images/train2017', 640, 2)
    val_loader = Yolo5Dataloader('../datasets/coco128/images/train2017', 640, 2)

    DT = DetectionTraining41C()
    DT.resume('runs/untitled_model.2/frozen.core', model, train_loader, val_loader, model.head.loss)

    # DT.debug_mode = True
    # DT.train(model, train_loader, val_loader, model.head.loss)

#     print(model.head.anchors.shape)

# #     data = LiveCamera(inf_size=416)
# #     det = DetectPipeline4C(model, class_names=coco_meta_data.names)

# #     det.stream(data, conf_thres=0.8, auto=True)


# from arch.data.detection_annotation import Yolo5Dataloader, coco_meta_data

# if __name__ == "__main__":
#     dataloader = Yolo5Dataloader("../datasets/coco128/images/train2017", image_size=640, batch_size=4, training=True)
#     anchor = dataloader.auto_anchor()
#     print(anchor)


# from arch.model import yolov5n
# from arch.data.detection_annotation import Yolo5Dataloader
# from arch.controller.detection_training_41c.core import DetectionTraining41C

# import torch

# if __name__ == '__main__':
#     model = yolov5n()
#     train_loader = Yolo5Dataloader('../datasets/coco128/images/train2017', 640, 4, training=True)
#     val_loader = Yolo5Dataloader('../datasets/coco128/images/train2017', 640, 4)
#     loss = model.head.loss
#     dt41c = DetectionTraining41C(model, train_loader, val_loader, loss)
#     dt41c.freeze('test.dt41c')
#     dt41c = DetectionTraining41C.resume('test.dt41c', train_loader, val_loader)
#     print(dt41c.model)