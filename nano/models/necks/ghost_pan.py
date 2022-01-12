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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        oc = out_channels[0]
        assert all([x == oc for x in out_channels])

        self.encoder = nn.ModuleList()
        for ic in in_channels:
            self.encoder.append(pointwise_conv(ic, oc))

        self.compressor_topdown = nn.ModuleList()
        for _ in in_channels[:-1]:
            self.compressor_topdown.append(GhostBlock(oc * 2, oc, 5))

        self.compressor_bottomup = nn.ModuleList()
        for _ in in_channels[1:]:
            self.compressor_bottomup.append(GhostBlock(oc * 2, oc, 5))

        self.downsample = nn.ModuleList()
        for _ in in_channels[1:]:
            self.downsample.append(dp_block(oc, 5, 2))


    def forward(self, xs):        
        # channel compression
        reshaped = [encoder(x) for encoder, x in zip(self.encoder, xs)]
        # top-down pathway
        for i in range(len(reshaped)-2, 0, -1):
            xcur = reshaped[i]
            xprev = reshaped[i+1]
            xcur = torch.cat([xcur, F.interpolate(xprev, scale_factor=2, mode="bilinear", align_corners=True)], dim=1)
            xcur = self.compressor_topdown[i](xcur)
            reshaped[i] = xcur
        # bottom-up pathway
        for i in range(len(reshaped)-1):
            xcur = reshaped[i+1]
            xprev = reshaped[i]
            xcur = torch.cat([xcur, self.downsample[i](xprev)], dim=1)
            xcur = self.compressor_bottomup[i](xcur)
            reshaped[i] = xcur
        return reshaped
