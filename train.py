import nano
import torch
import wandb

wandb.init(project="nano", dir="./runs")

# model setup
model = nano.models.yolox_esmk_shrink(num_classes=3)
model.load_state_dict(torch.load("runs/train/exp145/weights/best.pt", map_location="cpu")["state_dict"])
trainer = nano.detection.trainer

trainer.run(
    model=model,
    data="configs/coc-m.yaml",
    hyp="configs/hyp.finetune-nomixup.yaml",
    label_smoothing=0.1,
    batch_size=32,
    eval_batch_size=32,
    epochs=1000,
    imgsz=416,
    logger=wandb,
)
