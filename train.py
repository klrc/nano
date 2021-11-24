import nano
import torch
import wandb

wandb.init(project="nano", dir="./runs")

# model setup
model = nano.models.yolox_esmk_shrink_l(num_classes=3)
trainer = nano.detection.trainer

# speed run
trainer.run(
    model=model,
    data="configs/coc-l.yaml",
    ckpt="runs/train/exp158/weights/best.pt",
    hyp="configs/hyp.finetune-nomixup.yaml",
    logger=wandb,
    imgsz=416,
    # adam=True,
)
