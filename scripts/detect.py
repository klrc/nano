import nano

model = ...
imgsz = ...

detector = nano.detection.detector
detector.run(
    model,
    source="../datasets/tmpvoc",
    imgsz=imgsz,
    device="cpu",
)
