import torch.nn as nn
from .conv import pointwise_conv, depthwise_conv

# Inverted Residual module
# from mobilenetv2
# reference: https://blog.csdn.net/flyfish1986/article/details/97017017
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv1 = pointwise_conv(in_channels, mid_channels)
        self.conv2 = depthwise_conv(mid_channels, 3, 1)
        self.conv3 = pointwise_conv(mid_channels, out_channels, act=None)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x + identity
