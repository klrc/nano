import torch
import torch.nn as nn
import torch.nn.functional as F
from ..multiplex.conv import pointwise_conv, depthwise_conv
from ..multiplex.ghostnet import GhostBlock


# GhostPAN from NanoDet-plus
# https://github.com/RangiLyu/nanodet/blob/main/nanodet/model/fpn/ghost_pan.py
# https://zhuanlan.zhihu.com/p/449912627
class GhostPAN_3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pw_fs1 = pointwise_conv(in_channels[0], out_channels)
        self.pw_fs2 = pointwise_conv(in_channels[1], out_channels)
        self.pw_fs3 = pointwise_conv(in_channels[2], out_channels)

        self.in_csp_fs1 = GhostBlock(out_channels * 2, out_channels, 5)
        self.in_csp_fs2 = GhostBlock(out_channels * 2, out_channels, 5)

        self.out_csp_fs2 = GhostBlock(out_channels * 2, out_channels, 5)
        self.out_csp_fs3 = GhostBlock(out_channels * 2, out_channels, 5)

        self.dp_fs1 = nn.Sequential(
            depthwise_conv(out_channels, 5, 2),
            pointwise_conv(out_channels, out_channels),
        )
        self.dp_fs2 = nn.Sequential(
            depthwise_conv(out_channels, 5, 2),
            pointwise_conv(out_channels, out_channels),
        )

    def forward(self, x):
        fs1, fs2, fs3 = x
        # channel compression
        fs1, fs2, fs3 = self.pw_fs1(fs1), self.pw_fs2(fs2), self.pw_fs3(fs3)
        # top-down pathway
        fs3 = fs3
        fs2 = torch.cat([fs2, F.interpolate(fs3, scale_factor=2, mode="bilinear", align_corners=True)], dim=1)
        fs2 = self.in_csp_fs2(fs2)
        fs1 = torch.cat([fs1, F.interpolate(fs2, scale_factor=2, mode="bilinear", align_corners=True)], dim=1)
        fs1 = self.in_csp_fs1(fs1)
        # bottom-up pathway
        fs1 = fs1
        fs2 = torch.cat([fs2, self.dp_fs1(fs1)], dim=1)
        fs2 = self.out_csp_fs2(fs2)
        fs3 = torch.cat([fs3, self.dp_fs2(fs2)], dim=1)
        fs3 = self.out_csp_fs3(fs3)
        return [fs1, fs2, fs3]


class GhostPAN_4x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pw_fs0 = pointwise_conv(in_channels[0], out_channels)
        self.pw_fs1 = pointwise_conv(in_channels[1], out_channels)
        self.pw_fs2 = pointwise_conv(in_channels[2], out_channels)
        self.pw_fs3 = pointwise_conv(in_channels[3], out_channels)

        self.in_csp_fs0 = GhostBlock(out_channels * 2, out_channels, 5)
        self.in_csp_fs1 = GhostBlock(out_channels * 2, out_channels, 5)
        self.in_csp_fs2 = GhostBlock(out_channels * 2, out_channels, 5)

        self.out_csp_fs1 = GhostBlock(out_channels * 2, out_channels, 5)
        self.out_csp_fs2 = GhostBlock(out_channels * 2, out_channels, 5)
        self.out_csp_fs3 = GhostBlock(out_channels * 2, out_channels, 5)

        self.dp_fs0 = nn.Sequential(
            depthwise_conv(out_channels, 5, 2),
            pointwise_conv(out_channels, out_channels),
        )
        self.dp_fs1 = nn.Sequential(
            depthwise_conv(out_channels, 5, 2),
            pointwise_conv(out_channels, out_channels),
        )
        self.dp_fs2 = nn.Sequential(
            depthwise_conv(out_channels, 5, 2),
            pointwise_conv(out_channels, out_channels),
        )

    def forward(self, x):
        fs0, fs1, fs2, fs3 = x
        # channel compression
        fs0, fs1, fs2, fs3 = self.pw_fs0(fs0), self.pw_fs1(fs1), self.pw_fs2(fs2), self.pw_fs3(fs3)
        # top-down pathway
        fs3 = fs3
        fs2 = torch.cat([fs2, F.interpolate(fs3, scale_factor=2, mode="bilinear", align_corners=True)], dim=1)
        fs2 = self.in_csp_fs2(fs2)
        fs1 = torch.cat([fs1, F.interpolate(fs2, scale_factor=2, mode="bilinear", align_corners=True)], dim=1)
        fs1 = self.in_csp_fs1(fs1)
        fs0 = torch.cat([fs0, F.interpolate(fs1, scale_factor=2, mode="bilinear", align_corners=True)], dim=1)
        fs0 = self.in_csp_fs0(fs0)
        # bottom-up pathway
        fs0 = fs0
        fs1 = torch.cat([fs1, self.dp_fs0(fs0)], dim=1)
        fs1 = self.out_csp_fs1(fs1)
        fs2 = torch.cat([fs2, self.dp_fs1(fs1)], dim=1)
        fs2 = self.out_csp_fs2(fs2)
        fs3 = torch.cat([fs3, self.dp_fs2(fs2)], dim=1)
        fs3 = self.out_csp_fs3(fs3)
        return [fs1, fs2, fs3]