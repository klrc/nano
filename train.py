import nano


# model setup
model = nano.models.yolox_esmk_shrink_l(num_classes=3)
trainer = nano.detection.trainer

# speed-run
ckpt = trainer.run(
    model=model,
    data="configs/coc-s.yaml",
    hyp="configs/hyp.finetune-nomixup.yaml",
    # ckpt="runs/train/exp163/weights/last.pt",
    # load_optimizer=True,
    adam=True,
    patience=8,
    imgsz=416,
    epochs=50,
)

# finetune phase 2
ckpt = trainer.run(
    model=model,
    data="configs/coc-x.yaml",
    ckpt=ckpt,
    load_optimizer=False,
    hyp="configs/hyp.finetune-nomixup.yaml",
    patience=8,
    imgsz=416,
)

# finetune phase 3
ckpt = trainer.run(
    model=model,
    data="configs/coc-x.yaml",
    ckpt=ckpt,
    load_optimizer=False,
    hyp="configs/hyp.finetune-nomosaic.yaml",
    patience=8,
    imgsz=416,
)
