import nano
import torch
import wandb

wandb.init(project="nano", dir="./runs")

model = nano.models.yolox_shufflenetv2_es(num_classes=3)
# model.load_state_dict(torch.load("runs/train/exp120/weights/last.pt", map_location="cpu")["state_dict"])
trainer = nano.detection.trainer

trainer.run(
    model=model,
    data="configs/coc-s.yaml",
    hyp="configs/hyp.finetune.yaml",
    batch_size=32,
    eval_batch_size=32,
    epochs=1000,
    imgsz=416,
    adam=True,
    logger=wandb,
)
