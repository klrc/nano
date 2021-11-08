import nano
import torch

model = nano.models.yolox_cspm_depthwise_test(num_classes=3)
model.load_state_dict(torch.load("runs/train/exp109/weights/best.pt", map_location="cpu")["state_dict"])
detector = nano.detection.detector

detector.run(
    model,
    source="../datasets/fuh-testpic",
    imgsz=416,
    device="cpu",
)
