import nano
from nano._utils import freeze
import torch

model = nano.models.yolox_cspm_depthwise(num_classes=3)
# model.load_state_dict(torch.load("runs/train/exp112/weights/best.pt", map_location="cpu")["state_dict"])

freeze(model, "release/yolox_cspm_depthwise_3m/yolox_cspm.onnx", to_caffe=True, check_consistency=False)
print("anchors:", list(model.detect.anchor_grid.flatten().numpy()))
