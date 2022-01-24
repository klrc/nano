import torch
import torch.nn as nn
from .conv import pointwise_conv, depthwise_conv

class GhostBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.pw = pointwise_conv(in_channels, out_channels // 2)
        self.dw = depthwise_conv(out_channels // 2, kernel_size, 1, act=None)

    def forward(self, x):
        x = self.pw(x)
        x1 = self.dw(x)
        x = torch.cat([x1, x], dim=1)
        return x
