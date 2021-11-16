import nano
import torch
import wandb

wandb.init(project="nano", dir="./runs")

# from scratch
model = nano.models.yolox_esmk_shrink(num_classes=3)
trainer = nano.detection.trainer

trainer.run(
    model=model,
    data="configs/coc-s.yaml",
    hyp="configs/hyp.finetune-nomixup.yaml",
    label_smoothing=0.1,
    batch_size=32,
    eval_batch_size=32,
    epochs=1000,
    imgsz=416,
    adam=True,
    logger=wandb,
)
