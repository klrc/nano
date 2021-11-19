import nano
import torch
import wandb

wandb.init(project="nano", dir="./runs")

# model setup
model = nano.models.yolox_esmk_shrink_l(num_classes=3)
# ckpt = torch.load("runs/train/exp152/weights/best.pt", map_location="cpu")
# model.load_state_dict(ckpt["state_dict"])
trainer = nano.detection.trainer

# speed run
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
    # start_epoch=ckpt["epoch"],
    logger=wandb,
)
