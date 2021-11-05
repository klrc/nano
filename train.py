import nano
import torch
import wandb

wandb.init(project="nano", dir="./runs")

model = nano.models.yolox_cspm_depthwise_test(num_classes=3)
model.load_state_dict(torch.load("runs/train/exp101/weights/last.pt", map_location="cpu")["state_dict"])
trainer = nano.detection.trainer

trainer.run(
    model,
    data="configs/coc-l.yaml",
    hyp="configs/hyp.finetune.yaml",
    epochs=1000,
    imgsz=416,
    logger=wandb,
)
