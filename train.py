import nano
import torch

model = nano.models.yolox_cspm_depthwise_test(num_classes=3)
model.load_state_dict(torch.load("runs/train/exp100/weights/last.pt", map_location="cpu")["state_dict"])
trainer = nano.detection.trainer

trainer.run(
    model,
    data="configs/coc-l.yaml",
    hyp="configs/hyp.finetune.yaml",
    epochs=1000,
    imgsz=416,
)
