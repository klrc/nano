import nano
import wandb


# model setup
model = nano.models.yolox_esmk_shrink(num_classes=3)
trainer = nano.detection.trainer

# speed-run
ckpt = trainer.run(
    model=model,
    data="configs/coc-s.yaml",
    hyp="configs/hyp.finetune-nomixup.yaml",
    logger=wandb.init(project="nano", dir="./runs"),
    patience=32,
    imgsz=416,
    adam=True,
)

# finetune phase 1
ckpt = trainer.run(
    model=model,
    data="configs/coc-m.yaml",
    ckpt=ckpt,
    load_optimizer=False,
    hyp="configs/hyp.finetune-nomixup.yaml",
    logger=wandb.init(project="nano", dir="./runs"),
    patience=16,
    imgsz=416,
)

# finetune phase 2
ckpt = trainer.run(
    model=model,
    data="configs/coc-l.yaml",
    ckpt=ckpt,
    load_optimizer=False,
    hyp="configs/hyp.finetune-nomixup.yaml",
    logger=wandb.init(project="nano", dir="./runs"),
    patience=16,
    imgsz=416,
)

# finetune phase 3
ckpt = trainer.run(
    model=model,
    data="configs/coc-l.yaml",
    ckpt=ckpt,
    load_optimizer=False,
    hyp="configs/hyp.finetune-nomosaic.yaml",
    logger=wandb.init(project="nano", dir="./runs"),
    patience=16,
    imgsz=416,
)
