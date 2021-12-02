import torch
from torchvision.models import shufflenet_v2_x1_0

model = shufflenet_v2_x1_0()
torch.onnx.export(model, torch.rand(1, 3, 224, 416), "shufflenet-sample.onnx")
