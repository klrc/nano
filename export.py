import nano
from nano.export import freeze
import torch

model = nano.models.yolox_depthwise_cspm(num_classes=80)
# model.load_state_dict(torch.load("runs/train/exp100/weights/best.pt", map_location="cpu")["state_dict"])

freeze(model, "runs/build/yolox_depthwise_cspm_ez_test/yolox_cspm.onnx", to_caffe=True, check_consistency=False)
print("anchors:", list(model.detect.anchor_grid.flatten().numpy()))
