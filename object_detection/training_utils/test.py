import torch

from torchvision.models import resnet18

model = resnet18()
for k, v in model.named_parameters():
    print(k)