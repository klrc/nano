import torch
import nano
from nano.evolution import to_onnx, onnx_to_caffe

import torch.nn as nn

model = nano.models.yolov5_mobilenetv3_l()

onnx_path = to_onnx(model, "runs/build/test.onnx", dummy_input=(1, 1, 4, 4), output_names=["output"])
onnx_to_caffe(onnx_path)
