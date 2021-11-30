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


# MobileNetv2 Implementation ------------------------
# https://zhuanlan.zhihu.com/p/98874284


class Mv2Type1(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio):
        super().__init__()
        self.pw_1 = pointwise_conv(in_channels, in_channels * expand_ratio)
        self.dw = depthwise_conv(in_channels * expand_ratio, 3, 1)
        self.pw_2 = pointwise_conv(in_channels * expand_ratio, out_channels, act=None)
        self.use_res_connect = in_channels == out_channels

    def forward(self, x):
        identity = x
        x = self.pw_1(x)
        x = self.dw(x)
        x = self.pw_2(x)
        if self.use_res_connect:
            x += identity
        return x


class Mv2Type2(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio):
        super().__init__()
        self.pw_1 = pointwise_conv(in_channels, in_channels * expand_ratio)
        self.dw = depthwise_conv(in_channels * expand_ratio, 3, 2)
        self.pw_2 = pointwise_conv(in_channels * expand_ratio, out_channels, act=None)

    def forward(self, x):
        x = self.pw_1(x)
        x = self.dw(x)
        x = self.pw_2(x)
        return x


class MobileNetv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_s1 = nn.Sequential(
            nn.Sequential(nn.Conv2d(3, 16, 3, 2, bias=False), nn.BatchNorm2d(16), nn.ReLU6()),
            Mv2Type1(16, 16, 1),  # -
            Mv2Type2(16, 24, 6),  # -
            Mv2Type1(24, 24, 6),
            Mv2Type2(24, 40, 6),  # -
            Mv2Type1(40, 40, 6),
            Mv2Type1(40, 40, 6),
        )
        self.feature_s2 = nn.Sequential(
            Mv2Type2(40, 64, 6),  # -
            Mv2Type1(64, 64, 6),
            Mv2Type1(64, 64, 6),
            Mv2Type1(64, 64, 6),
            Mv2Type1(64, 80, 6),  # -
            Mv2Type1(80, 80, 6),
            Mv2Type1(80, 80, 6),
        )
        self.feature_s3 = nn.Sequential(
            Mv2Type2(80, 160, 6),  # -
            Mv2Type1(160, 160, 6),
            Mv2Type1(160, 160, 6),
            Mv2Type1(160, 160, 6),  # -
        )

    def forward(self, x):
        fs1 = self.feature_s1(x)
        fs2 = self.feature_s2(fs1)
        fs3 = self.feature_s3(fs2)
        return [fs1, fs2, fs3]


# ---------------------------------------------------


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


# Enhanced ShuffleNet block with stride=2
# Arxiv: https://arxiv.org/pdf/2111.00902.pdf
class ESBlockS2(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.part1 = nn.Sequential(
            pointwise_conv(in_channels, mid_channels),
            depthwise_conv(mid_channels, 3, 2, act=None),
            pointwise_conv(mid_channels, out_channels // 2),
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
class CSPPAN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c3_conv = pointwise_conv(in_channels[0], out_channels)
        self.c4_conv = pointwise_conv(in_channels[1], out_channels)
        self.c5_conv = pointwise_conv(in_channels[2], out_channels)
        # self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.c3_csp = CSPBlock(out_channels * 2, out_channels)
        self.c4_csp_1 = CSPBlock(out_channels * 2, out_channels)
        self.c4_csp_2 = CSPBlock(out_channels * 2, out_channels)
        self.c5_csp = CSPBlock(out_channels * 2, out_channels)
        self.downsample_1 = nn.Sequential(
            depthwise_conv(out_channels, 5, 2),
            pointwise_conv(out_channels, out_channels),
        )
        self.downsample_2 = nn.Sequential(
            depthwise_conv(out_channels, 5, 2),
            pointwise_conv(out_channels, out_channels),
        )

    def forward(self, x):
        c3, c4, c5 = x
        c3 = self.c3_conv(c3)  # channel down-scale
        c4 = self.c4_conv(c4)
        c5 = self.c5_conv(c5)
        c4 = torch.cat([c4, F.interpolate(c5, scale_factor=2, mode="bilinear", align_corners=True)], dim=1)
        c4 = self.c4_csp_1(c4)
        c3 = torch.cat([c3, F.interpolate(c4, scale_factor=2, mode="bilinear", align_corners=True)], dim=1)
        c3 = self.c3_csp(c3)
        c4 = torch.cat([c4, self.downsample_1(c3)], dim=1)
        c4 = self.c4_csp_2(c4)
        c5 = torch.cat([c5, self.downsample_2(c4)], dim=1)
        c5 = self.c5_csp(c5)
        return [c3, c4, c5]


# Detection Head modified from yolov5 series
# Ultralytics version
class DetectHead(nn.Module):
    def __init__(self, in_channels, num_classes, anchors):  # detection layer
        super().__init__()
        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.stride = torch.tensor((8, 16, 32))  # strides computed during build
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)  # shape(nl,na,2)
        self.register_buffer("anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.mode_dsp_off = True
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in in_channels)  # output conv

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            if self.mode_dsp_off:
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
        if self.training:
            return x
        else:
            return torch.cat(z, 1), x


# Full detection model
class M2yolov5(nn.Module):
    def __init__(self, num_classes, anchors):
        super().__init__()
        self.backbone = MobileNetv2()
        self.neck = CSPPAN([40, 80, 160], 80)
        self.head = DetectHead([80, 80, 80], num_classes, anchors)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


def mobilenet_v2_cspp_yolov5(
    num_classes=3,
    anchors=(
        [5, 6, 10, 13, 16, 30, 33, 23],
        [15, 30, 30, 61, 62, 45, 59, 119],
        [58, 45, 116, 90, 156, 198, 373, 326],
    ),
):
    return M2yolov5(num_classes=num_classes, anchors=anchors)


if __name__ == "__main__":
    model = mobilenet_v2_cspp_yolov5()
    for y in model(torch.rand(4, 3, 224, 416)):
        print(y.shape)
