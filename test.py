import nano
import torch

from nano.datasets.object_detection import evaluator

model = nano.models.yolox_cspm()
model.load_state_dict(torch.load('runs/train/exp50/weights/best.pt', map_location='cpu')['state_dict'])
trainer = nano.datasets.object_detection.evaluator

evaluator.run(
    model,
    data="nano/datasets/object_detection/configs/coc-n.yaml",
    imgsz=416,
)


# 11,13, 28,46, 143,120