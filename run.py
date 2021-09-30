from nano.datasets.object_detection import trainer
from nano.models.yolov5_shufflenet_1_5x import yolov5_shufflenet_1_5x

model = yolov5_shufflenet_1_5x(num_classes=80)
trainer.run(model)