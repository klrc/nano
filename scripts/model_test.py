import torch
import nano
import yaml
from nano.detection import CaffeWrapper, evaluator


model = nano.models.esnet_cspp_yolov5_s3__seblock_canceled()
model.load_state_dict(torch.load("runs/train/exp190/weights/best.pt", map_location="cpu")["state_dict"])
model.names = {0: "person", 1: "bike", 2: "car"}
# model.eval()

evaluator = nano.detection.evaluator
evaluator.run(
    model,
    # data="configs/coco-val.yaml",
    data="configs/coc-x.yaml",
    batch_size=32,
    imgsz=416,
    device="0",
)
