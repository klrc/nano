from nano.datasets.object_detection import trainer

# from nano.models.yolov5_shufflenet_1_5x import yolov5_shufflenet_1_5x
# model = yolov5_shufflenet_1_5x(num_classes=80, backbone_ckpt='nano/models/yolov5_shufflenet_1_5x/nanodet_m_1.5x_416.ckpt')

from nano.models.deprecated_yolov5s.models import load_model
model = load_model('/home/sh/Projects/klrc/yolov5s-trainer/runs/train/exp90/weights/best.pt', 'cpu', 6)[0]

trainer.run(model, data='data/coco-s.yaml', imgsz=416)
