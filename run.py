from nano.datasets.object_detection import trainer

from nano.models.yolov5_shufflenet_1_5x import yolov5_shufflenet_1_5x
model = yolov5_shufflenet_1_5x(num_classes=6, backbone_ckpt='nano/models/yolov5_shufflenet_1_5x/nanodet_m_1.5x_416.ckpt')

trainer.run(model, data='data/coco-s.yaml', hyp='data/hyps/hyp.finetune.yaml', epochs=1000, imgsz=416)


# autoanchor: Evolving anchors with Genetic Algorithm: fitness = 0.6985: 100%|████████████████████████████████████████████| 1000/1000 [00:21<00:00, 45.90it/s]
# autoanchor: thr=0.34: 0.9903 best possible recall, 3.12 anchors past thr
# autoanchor: n=9, img_size=416, metric_all=0.292/0.701-mean/best, past_thr=0.537-mean: 6,7,  8,19,  18,14,  16,36,  39,29,  31,77,  83,61,  75,159,  223,179
# autoanchor: New anchors saved to model. Update model *.yaml to use these anchors in the future.
