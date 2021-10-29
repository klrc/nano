import nano
from nano.evolution import freeze
import torch

model = nano.models.yolox_cspm()
model.load_state_dict(torch.load("runs/train/exp63/weights/best.pt", map_location="cpu")["state_dict"])

freeze(model, "runs/build/yolox_depthwise_cspm/yolox_cspm.onnx", to_caffe=True, check_consistency=False)
print("anchors:", list(model.detect.anchor_grid.flatten().numpy()))
