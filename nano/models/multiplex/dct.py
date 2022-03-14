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


class DCTConv2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int, ratio_y=0.9, ratio_cb=0.5, ratio_cr=0.5) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        for ratio in (ratio_y, ratio_cb, ratio_cr):
            tri_len = int(kernel_size * ratio)
            out_channels = int((tri_len + 1) * tri_len / 2)
            conv = nn.Conv2d(1, out_channels, kernel_size, stride, bias=False)
            conv.weight.data = self.dct_weight(kernel_size, ratio)
            self.convs.append(conv)
        self.out_channels = sum([x.weight.data.size(0) for x in self.convs])

    def dct_weight(self, kernel_size, ratio=1):
        # get sampled dct pairs
        dct_index = []
        dct_size = int(kernel_size * ratio)  # ratio = 1 for full-channel DCT
        for i in range(dct_size):
            for j in range(dct_size - i):  # top-left triangular matrix (sort by frequency)
                dct_index.append((i, j))
        # create preset DCT weights
        weight = torch.zeros(len(dct_index), 1, kernel_size, kernel_size)
        for i in range(kernel_size):
            for j in range(kernel_size):
                for ii, (d_i, d_j) in enumerate(dct_index):
                    x_a = np.cos(d_i * np.pi / kernel_size * (i + 0.5))
                    x_b = np.cos(d_j * np.pi / kernel_size * (j + 0.5))
                    weight[ii, 0, i, j] = x_a * x_b
        return weight

    def forward(self, x: torch.Tensor):
        xs = []
        for i in range(3):
            xs.append(self.convs[i](x[:, i : i + 1]))
        return torch.cat(xs, dim=1)


class DCTModule(nn.Module):
    def __init__(self, kernel_size, stride) -> None:
        super().__init__()
        self.rgb2ycbcr = RGB2YCbCrConv2d()
        self.conv = DCTConv2d(kernel_size, stride)
        # self.norm = nn.GroupNorm(num_groups=self.conv.out_channels, num_channels=self.conv.out_channels)
        # for p in self.rgb2ycbcr.parameters():
        #     p.requires_grad = False
        # for p in self.conv.parameters():
        #     p.requires_grad = False

    def forward(self, x):
        x = self.rgb2ycbcr(x)
        x = self.conv(x)
        return x
