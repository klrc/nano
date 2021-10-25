import nano
from nano.evolution import freeze
import torch

model = nano.models.yolox_cspm()
model.load_state_dict(torch.load('runs/train/exp50/weights/best.pt', map_location='cpu')['state_dict'])

freeze(model, "runs/build/yolox_cspm/yolox_cspm.onnx", to_caffe=True)
