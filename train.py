import nano
import torch

model = nano.models.yolox_cspm()
model.load_state_dict(torch.load('runs/train/exp51/weights/last.pt', map_location='cpu')['state_dict'])
trainer = nano.datasets.object_detection.trainer

trainer.run(
    model,
    data="nano/datasets/object_detection/configs/coc-n.yaml",
    hyp="nano/datasets/object_detection/configs/hyps/hyp.finetune.yaml",
    epochs=1000,
    imgsz=416,
)

# 20,23, 70,93, 237,202