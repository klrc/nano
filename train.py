from nano.datasets.object_detection import trainer
import torch

from nano.models.yolov5_mobilenetv3_l import yolov5_mobilenetv3_l
model = yolov5_mobilenetv3_l(num_classes=6)
model.load_state_dict(torch.load('release/yolov5_mobilenetv3_l@coco-s/best.pt')['state_dict'])

trainer.run(model, data='data/coco-s+indoor.yaml', hyp='data/hyps/hyp.finetune.yaml', epochs=1000, imgsz=416)
