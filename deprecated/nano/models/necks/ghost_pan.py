import enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..multiplex.conv import pointwise_conv, depthwise_conv
from ..multiplex.ghostnet import GhostBlock


def dp_block(channels, kernel_size, stride):
    return nn.Sequential(
        depthwise_conv(channels, kernel_size, stride),
        pointwise_conv(channels, channels),
    )


# GhostPAN from NanoDet-plus
# https://github.com/RangiLyu/nanodet/blob/main/nanodet/model/fpn/ghost_pan.py
# https://zhuanlan.zhihu.com/p/449912627
class GhostPAN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_branches):
        super().__init__()
        in_branches = len(in_channels)
        self.in_branches = in_branches
        self.out_branches = out_branches
        # reshape module
        self.reshape = nn.ModuleList()
        for ic in in_channels:
            self.reshape.append(pointwise_conv(ic, hidden_channels))
        # compressor for top-down pathway
        self.compressor_upsample = nn.ModuleList()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        for _ in range(in_branches - 1):
            self.compressor_upsample.append(GhostBlock(hidden_channels * 2, hidden_channels, 5))
        # compressor for bottom-up pathway
        self.compressor_downsample = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for _ in range(in_branches - 1):
            self.compressor_downsample.append(GhostBlock(hidden_channels * 2, hidden_channels, 5))
            self.downsample.append(dp_block(hidden_channels, 5, 2))
        # additional P6
        if out_branches > in_branches:
            assert out_branches == in_branches + 1
            # two additional downsample module
            for _ in range(2):
                self.downsample.append(dp_block(hidden_channels, 5, 2))

    def forward(self, xs):
        # channel compression
        xs = [encoder(x) for encoder, x in zip(self.reshape, xs)]
        # top-down pathway (upsample)
        N = self.in_branches
        for i in range(N - 1):
            _xcur, _next = xs[N - 1 - i], xs[N - 2 - i]
            _next = torch.cat([self.upsample(_xcur), _next], dim=1)
            _next = self.compressor_upsample[i](_next)
            xs[N - 2 - i] = _next
        # bottom-up pathway (downsample)
        N = self.out_branches
        if N == 3:
            P3 = xs[0]
            P4 = torch.cat([self.downsample[0](P3), xs[1]], dim=1)
            P4 = self.compressor_downsample[0](P4)
            P5 = torch.cat([self.downsample[1](P4), xs[2]], dim=1)
            P5 = self.compressor_downsample[1](P5)
            return P3, P4, P5
        elif N == 4:
            P3 = xs[0]
            P4 = torch.cat([self.downsample[0](P3), xs[1]], dim=1)
            P4 = self.compressor_downsample[0](P4)
            P5_DP = self.downsample[-1](xs[-1])
            P5 = torch.cat([self.downsample[1](P4), xs[2]], dim=1)
            P5 = self.compressor_downsample[1](P5)
            P6 = self.downsample[-2](P5) + P5_DP
            return P3, P4, P5, P6
        else:
            raise NotImplementedError
