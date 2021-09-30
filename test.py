
# Trainer Scripts

from nano.datasets.object_detection import trainer, evaluator
from nano.models.yolov5_shufflenet_1_5x import yolov5_shufflenet_1_5x

model = yolov5_shufflenet_1_5x(nc=80)
trainer.run(model, data='coco-128.yaml')