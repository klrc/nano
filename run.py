from nano.datasets.object_detection import trainer
import torch

from nano.models.yolov5_shufflenet_1_5x import yolov5_shufflenet_1_5x
model = yolov5_shufflenet_1_5x(num_classes=6, backbone_ckpt='nano/models/yolov5_shufflenet_1_5x/nanodet_m_1.5x_416.ckpt')
model.load_state_dict(torch.load('runs/train/exp22/weights/last.pt')['state_dict'])

trainer.run(model, data='data/coco-x.yaml', hyp='data/hyps/hyp.finetune.yaml', epochs=1000, imgsz=416)
