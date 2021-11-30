import nano


# model setup
model = nano.models.mobilenet_v2_cspp_yolov5(num_classes=3)
trainer = nano.detection.trainer

# speed-run
ckpt = trainer.run(
    model=model,
    data="configs/coc-s.yaml",
    hyp="configs/hyp.finetune-nomixup.yaml",
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
    patience=4,
    imgsz=416,
)
