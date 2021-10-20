
import nano
import torch

model = nano.models.yolov5_mobilenetv3_l(num_classes=6)
model.load_state_dict(torch.load('/Volumes/ASM236X/best.pt', map_location='cpu')['state_dict'])
detector = nano.datasets.object_detection.detector

detector.run(model, source='/Volumes/ASM236X/fuh-testpic', imgsz=416)