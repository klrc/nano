import nano
import torch

model = ...
detector = nano.detection.detector

detector.run(
    model,
    source="../datasets/fuh-testpic",
    imgsz=416,
    device="cpu",
)
