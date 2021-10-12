from nano.datasets.object_detection import trainer
import torch

from nano.models.yolov5_mobilenetv3_s import yolov5_mobilenetv3_s
model = yolov5_mobilenetv3_s(num_classes=6)
model.forward(torch.rand(4, 3, 224, 416))

# trainer.run(model, data='data/coco-s+indoor.yaml', hyp='data/hyps/hyp.finetune.yaml', epochs=1000, imgsz=416)
