import nano
from nano._utils import freeze
import torch

model = nano.models.yolox_esmk_shrink_misc(num_classes=4).dsp()
# model.load_state_dict(torch.load("runs/train/exp130/weights/best.pt", map_location="cpu")["state_dict"])
output_names = ["output_1", "output_2", "output_3"]

freeze(
    model,
    onnx_path="release/yolox_esmk_misc/yolox_esmk_shrink.onnx",
    to_caffe=True,
    check_consistency=False,
    output_names=output_names,
)
print("class_names:", ["person", "bike", "car"])
print("anchors:", list(model.head.anchor_grid.flatten().numpy()))
print("output_names:", output_names)
print('misc_thres:', model.head.misc_bias.view(1).item())
