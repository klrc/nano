import nano
import torch

model = nano.models.yolox_esmk_shrink(num_classes=4)
model.load_state_dict(torch.load("runs/train/exp130/weights/best.pt", map_location="cpu")["state_dict"])
detector = nano.detection.detector

detector.run(
    model,
    source="../datasets/fuh-testpic",
    imgsz=416,
    device="cpu",
)
