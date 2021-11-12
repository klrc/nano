import nano
from nano._utils import freeze
import torch

model = nano.models.yolox_esmk_shrink(num_classes=3)
# model.load_state_dict(torch.load("runs/train/exp112/weights/best.pt", map_location="cpu")["state_dict"])

freeze(
    model,
    onnx_path="release/yolox_esmk/yolox_esmk_shrink.onnx",
    to_caffe=True,
    check_consistency=False,
)
# print("anchors:", list(model.detect.anchor_grid.flatten().numpy()))
