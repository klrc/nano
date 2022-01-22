import torch
import torch.nn as nn
from ..multiplex.conv import pointwise_conv, depthwise_conv
from ..multiplex.ghostnet import GhostBlock

# ESNet Implementation ------------------------------
# https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/ppdet/modeling/backbones/esnet.py
# Arxiv: https://arxiv.org/pdf/2111.00902.pdf


def make_divisible(v, divisor=16, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# Enhanced ShuffleNet block with stride=2
class ESBlockS2(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.part1 = nn.Sequential(
            pointwise_conv(in_channels, mid_channels // 2),
            depthwise_conv(mid_channels // 2, 3, 2, act=None),
            pointwise_conv(mid_channels // 2, out_channels // 2),
        )
        self.part2 = nn.Sequential(
            depthwise_conv(in_channels, 3, 2, act=None),
            pointwise_conv(in_channels, out_channels // 2),
        )
        self.dp = nn.Sequential(
            depthwise_conv(out_channels, 3, 1),
            pointwise_conv(out_channels, out_channels),
        )

    def forward(self, x):
        x1 = self.part1(x)
        x2 = self.part2(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.dp(x)
        return x


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups, channels_per_group, height, width)  # reshape
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)  # flatten
        return x


# Enhanced ShuffleNet block with stride=1
class ESBlockS1(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.main_branch = nn.Sequential(
            GhostBlock(in_channels // 2, mid_channels),
            pointwise_conv(mid_channels, in_channels // 2),
        )
        self.split_point = in_channels // 2
        self.channel_shuffle = ChannelShuffle(2)

    def forward(self, x):
        x1, x2 = x[:, : self.split_point, :, :], x[:, self.split_point :, :, :]
        x1 = self.main_branch(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.channel_shuffle(x)
        return x


class EnhancedShuffleNetv2_4x(nn.Module):
    def __init__(self, channels=(24, 48, 96, 192, 288)):
        super().__init__()
        channels = [x if i==0 else make_divisible(x) for i, x in enumerate(channels)]

        __inc, __mid = channels[0], channels[1]
        self.feature_s0 = nn.Sequential(
            nn.Conv2d(3, __inc, 3, 2, 1),
            nn.BatchNorm2d(__inc),
            nn.ReLU6(),
            ESBlockS2(__inc, __mid, __mid),
            ESBlockS1(__mid, __mid),
            ESBlockS1(__mid, __mid),
        )
        __inc, __mid = channels[1], channels[2]
        self.feature_s1 = nn.Sequential(
            ESBlockS2(__inc, __mid, __mid),
            ESBlockS1(__mid, __mid),
            ESBlockS1(__mid, __mid),
        )
        __inc, __mid = channels[2], channels[3]
        self.feature_s2 = nn.Sequential(
            ESBlockS2(__inc, __mid, __mid),
            ESBlockS1(__mid, __mid),
            ESBlockS1(__mid, __mid),
            ESBlockS1(__mid, __mid),
            ESBlockS1(__mid, __mid),
            ESBlockS1(__mid, __mid),
            ESBlockS1(__mid, __mid),
        )
        __inc, __mid = channels[3], channels[4]
        self.feature_s3 = nn.Sequential(
            ESBlockS2(__inc, __mid, __mid),
            ESBlockS1(__mid, __mid),
            ESBlockS1(__mid, __mid),
        )

    def forward(self, x):
        fs0 = self.feature_s0(x)
        fs1 = self.feature_s1(fs0)
        fs2 = self.feature_s2(fs1)
        fs3 = self.feature_s3(fs2)
        return [fs0, fs1, fs2, fs3]


class EnhancedShuffleNetv2_3x(nn.Module):
    def __init__(self, channels=(24, 96, 192, 384)):
        super().__init__()
        channels = [x if i==0 else make_divisible(x) for i, x in enumerate(channels)]

        __inc = channels[0]
        self.feature_s0 = nn.Sequential(
            nn.Conv2d(3, __inc, 3, 2, 1),
            nn.BatchNorm2d(__inc),
            nn.ReLU6(),
            nn.MaxPool2d(3, 2, 1),
        )
        __inc, __mid = channels[0], channels[1]
        self.feature_s1 = nn.Sequential(
            ESBlockS2(__inc, __mid, __mid),
            ESBlockS1(__mid, __mid),
            ESBlockS1(__mid, __mid),
        )
        __inc, __mid = channels[1], channels[2]
        self.feature_s2 = nn.Sequential(
            ESBlockS2(__inc, __mid, __mid),
            ESBlockS1(__mid, __mid),
            ESBlockS1(__mid, __mid),
            ESBlockS1(__mid, __mid),
            ESBlockS1(__mid, __mid),
            ESBlockS1(__mid, __mid),
            ESBlockS1(__mid, __mid),
        )
        __inc, __mid = channels[2], channels[3]
        self.feature_s3 = nn.Sequential(
            ESBlockS2(__inc, __mid, __mid),
            ESBlockS1(__mid, __mid),
            ESBlockS1(__mid, __mid),
        )

    def forward(self, x):
        fs0 = self.feature_s0(x)
        fs1 = self.feature_s1(fs0)
        fs2 = self.feature_s2(fs1)
        fs3 = self.feature_s3(fs2)
        return [fs1, fs2, fs3]