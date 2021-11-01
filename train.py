import nano
import torch

model = nano.models.yolox_depthwise_cspm(num_classes=3)
# model.load_state_dict(torch.load('runs/train/exp61/weights/best.pt', map_location='cpu')['state_dict'])
trainer = nano.datasets.object_detection.trainer

trainer.run(
    model,
    data="nano/datasets/object_detection/configs/coc-s.yaml",
    hyp="nano/datasets/object_detection/configs/hyps/hyp.scratch.yaml",
    epochs=1000,
    imgsz=416,
    label_smoothing=0.001,
)

