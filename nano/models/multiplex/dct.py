import torch
import torch.nn as nn
import numpy as np


class RGB2YCbCrConv2d(nn.Conv2d):
    # transform RGB image to Y, Cb, Cr space
    def __init__(self) -> None:
        super().__init__(in_channels=3, out_channels=3, kernel_size=1)
        weight, bias = self.rgb2ycbcr_weight()
        self.weight.data = weight
        self.bias.data = bias

    def rgb2ycbcr_weight(self):
        # using formulas from https://en.wikipedia.org/wiki/YCbCr
        weight = [
            [65.481, 128.553, 24.966],
            [-37.797, -74.203, 112.0],
            [112.0, -93.786, -18.214],
        ]
        bias = [16.0, 128.0, 128.0]
        weight = torch.tensor(weight).unsqueeze(-1).unsqueeze(-1)  # (out, in, k, k)
        bias = torch.tensor(bias)
        return weight, bias


class DCTConv2d(nn.Conv2d):
    def __init__(self, kernel_size: int):
        super().__init__(in_channels=3, out_channels=kernel_size ** 2 * 3, kernel_size=kernel_size, stride=kernel_size, bias=False, groups=3)
        self.weight.data = self.dct_weight(kernel_size)

    def dct_weight(self, kernel_size, input_channels=3):
        dct_size = kernel_size  # full-channel DCT
        weight = torch.zeros(dct_size ** 2 * input_channels, 1, kernel_size, kernel_size)
        for i in range(kernel_size):
            for j in range(kernel_size):
                for d_i in range(dct_size):
                    for d_j in range(dct_size):
                        for input_channel in range(input_channels):
                            x_a = np.cos(d_i * np.pi / dct_size * (i + 0.5))
                            x_b = np.cos(d_j * np.pi / dct_size * (j + 0.5))
                            weight[input_channel * dct_size * dct_size + d_i * dct_size + d_j, 0, i, j] = x_a * x_b
        return weight


class ChannelGate(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.gate = nn.Parameter(data=torch.zeros(channels).view(1, -1, 1, 1), requires_grad=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        g = self.softmax(self.gate)
        return x * g


class DCTModule(nn.Module):
    def __init__(self, kernel_size) -> None:
        super().__init__()
        self.rgb2ycbcr = RGB2YCbCrConv2d()
        self.conv = DCTConv2d(kernel_size)
        self.norm = nn.GroupNorm(num_groups=kernel_size ** 2 * 3, num_channels=kernel_size ** 2 * 3)
        self.gate = ChannelGate(kernel_size ** 2 * 3)
        for p in self.rgb2ycbcr.parameters():
            p.requires_grad = False
        for p in self.conv.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.rgb2ycbcr(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.gate(x)
        return x
