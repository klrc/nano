import torch
import torch.nn as nn
import torch.nn.functional as F

__default_norm = nn.BatchNorm2d
__default_activation = nn.ReLU6


# initialize parameters
def init_parameters(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)


# basic conv+norm+relu combination
def _conv_norm_act(
    in_channels, out_channels, kernel_size, stride, padding, groups, norm=__default_norm, act=__default_activation
):
    layers = []
    layers.append(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=norm is None,
        )
    )
    if norm is not None:
        layers.append(norm(out_channels))
    if act is not None:
        layers.append(act())
    return nn.Sequential(*layers)


# pointwise convolution
def pointwise_conv(in_channels, out_channels, norm=__default_norm, act=__default_activation):
    return _conv_norm_act(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        groups=1,
        norm=norm,
        act=act,
    )


# depthwise convolution
def depthwise_conv(in_channels, kernel_size, stride, norm=__default_norm, act=__default_activation):
    return _conv_norm_act(
        in_channels=in_channels,
        out_channels=in_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=(kernel_size - 1) // 2,
        groups=in_channels,
        norm=norm,
        act=act,
    )


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


class GhostBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pw = pointwise_conv(in_channels, out_channels // 2)
        self.dw = depthwise_conv(out_channels // 2, 3, 1, act=None)

    def forward(self, x):
        x = self.pw(x)
        x1 = self.dw(x)
        x = torch.cat([x1, x], dim=1)
        return x


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


class EnhancedShuffleNetv2(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [3, 64, 128, 256, 384]
        __inc = channels.pop(0)
        __mid = channels.pop(0)
        self.feature_s0 = nn.Sequential(
            _conv_norm_act(__inc, __mid, 3, 2, 1, 1),
            nn.MaxPool2d(3, 2, 1),
            ESBlockS1(__mid, __mid),
            ESBlockS1(__mid, __mid),
            ESBlockS1(__mid, __mid),
        )
        __inc = __mid
        __mid = channels.pop(0)
        self.feature_s1 = nn.Sequential(
            ESBlockS2(__inc, __mid, __mid),
            ESBlockS1(__mid, __mid),
            ESBlockS1(__mid, __mid),
        )
        __inc = __mid
        __mid = channels.pop(0)
        self.feature_s2 = nn.Sequential(
            ESBlockS2(__inc, __mid, __mid),
            ESBlockS1(__mid, __mid),
            ESBlockS1(__mid, __mid),
            ESBlockS1(__mid, __mid),
            ESBlockS1(__mid, __mid),
            ESBlockS1(__mid, __mid),
            ESBlockS1(__mid, __mid),
        )
        __inc = __mid
        __mid = channels.pop(0)
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


# ---------------------------------------------------

# ================================================================================================
# Main Structure =================================================================================
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


# Cross Stage Partial Network block
# fusion-first mode
# CVF: https://openaccess.thecvf.com/content_CVPRW_2020/papers/w28/Wang_CSPNet_A_New_Backbone_That_Can_Enhance_Learning_Capability_of_CVPRW_2020_paper.pdf
class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.short_conv = pointwise_conv(in_channels, in_channels // 2)
        self.main_conv = pointwise_conv(in_channels, in_channels // 2)
        self.dense_block = InvertedResidual(in_channels // 2, in_channels, in_channels // 2)
        self.post_transition = pointwise_conv(in_channels, out_channels)

    def forward(self, x):
        x1 = self.short_conv(x)
        x2 = self.main_conv(x)
        x2 = self.dense_block(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.post_transition(x)
        return x


# CSPPAN from NanoDet series
# reference: https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/ppdet/modeling/heads/pico_head.py
class CSPPANS3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pw_fs1 = pointwise_conv(in_channels[0], out_channels)
        self.pw_fs2 = pointwise_conv(in_channels[1], out_channels)
        self.pw_fs3 = pointwise_conv(in_channels[2], out_channels)

        self.in_csp_fs1 = CSPBlock(out_channels * 2, out_channels)
        self.in_csp_fs2 = CSPBlock(out_channels * 2, out_channels)

        self.out_csp_fs2 = CSPBlock(out_channels * 2, out_channels)
        self.out_csp_fs3 = CSPBlock(out_channels * 2, out_channels)

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



class CSPPANS4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pw_fs0 = pointwise_conv(in_channels[0], out_channels)
        self.pw_fs1 = pointwise_conv(in_channels[1], out_channels)
        self.pw_fs2 = pointwise_conv(in_channels[2], out_channels)
        self.pw_fs3 = pointwise_conv(in_channels[3], out_channels)

        self.in_csp_fs0 = CSPBlock(out_channels * 2, out_channels)
        self.in_csp_fs1 = CSPBlock(out_channels * 2, out_channels)
        self.in_csp_fs2 = CSPBlock(out_channels * 2, out_channels)

        self.out_csp_fs1 = CSPBlock(out_channels * 2, out_channels)
        self.out_csp_fs2 = CSPBlock(out_channels * 2, out_channels)
        self.out_csp_fs3 = CSPBlock(out_channels * 2, out_channels)

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

class ReIDHead(nn.Module):
    def __init__(self, in_channels, reid_channels, stages=3):
        super().__init__()
        self.m = nn.ModuleList(nn.Conv2d(in_channels, reid_channels, 1) for _ in range(stages))  # output conv

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            if not self.training:
                y = x[i].flatten(1)
                y = y.div(y.norm(p=2, dim=1, keepdim=True))
                z.append(y)
        if self.training:
            return x
        else:
            return torch.cat(z, 1), x


# Detection Head modified from yolov5 series
# Ultralytics version
class AnchorsHead(nn.Module):
    def __init__(self, in_channels, num_classes, anchors, strides=(8, 16, 32)):  # detection layer
        super().__init__()
        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.stride = torch.tensor(strides)  # strides computed during build
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)
        self.register_buffer("anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(in_channels, self.no * self.na, 1) for _ in range(len(strides)))  # output conv
        self.dsp_mode = False

    def dsp(self):
        self.dsp_mode = True

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            if not self.dsp_mode:
                # reshape
                bs, _, ny, nx = x[i].shape
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                if not self.training:
                    # make grid
                    if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
                        self.grid[i] = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float().to(x[i].device)
                    # normalize and get xywh
                    y = x[i].sigmoid()
                    y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    z.append(y.view(bs, -1, self.no))
        if self.training or self.dsp_mode:
            return x
        else:
            return torch.cat(z, 1), x


class YoloDefense(nn.Module):
    def __init__(self, num_classes, anchors, reid_channels):
        super().__init__()
        self.backbone = EnhancedShuffleNetv2()
        self.neck = CSPPANS4([64, 128, 256, 384], 96)
        self.head = AnchorsHead(96, num_classes, anchors)
        self.reid_head = ReIDHead(96, reid_channels)
        self.reid_mode = False

    def switch_mode(self, reid=True):
        self.reid_mode = reid

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        if self.reid_mode:
            return self.reid_head(x)
        else:
            return self.head(x)


def yolo_defense_es_96h_4x(
    num_classes=3,
    anchors=(
        [10, 13, 16, 30, 33, 23],
        [10, 13, 16, 30, 33, 23],
        [10, 13, 16, 30, 33, 23],
    ),
):
    model = YoloDefense(num_classes=num_classes, anchors=anchors, reid_channels=64)
    return model


if __name__ == "__main__":
    model = yolo_defense_es_96h_4x()
    model.train()

    params = 0
    for module in model.modules():
        if hasattr(module, "weight") and hasattr(module.weight, "size"):
            params += torch.prod(torch.LongTensor(list(module.weight.size())))
        if hasattr(module, "bias") and hasattr(module.bias, "size"):
            params += torch.prod(torch.LongTensor(list(module.bias.size())))
    total_params_size = abs(params.numpy() * 4. / (1024 ** 2.))
    print("Params size (MB): %0.2f" % total_params_size)

    print("init finished")
    pred = model(torch.rand(4, 3, 224, 416))
    for y in pred:
        print(y.shape)
