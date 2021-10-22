
import nano

model = nano.models.yolov5_cspm()
trainer = nano.datasets.object_detection.trainer

# trainer.run(model, data='data/coco-s+indoor.yaml', hyp='data/hyps/hyp.finetune.yaml', epochs=1000, imgsz=416)