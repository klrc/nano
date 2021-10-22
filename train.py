import nano

model = nano.models.yolov5_cspm(num_classes=3)
trainer = nano.datasets.object_detection.trainer

trainer.run(
    model,
    data="nano/datasets/object_detection/configs/coc-s.yaml",
    hyp="nano/datasets/object_detection/configs/hyps/hyp.scratch.yaml",
    epochs=1000,
    imgsz=416,
)
