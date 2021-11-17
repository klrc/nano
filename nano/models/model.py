import torch
import torch.nn as nn
import torch.nn.functional as F

__default_norm = nn.BatchNorm2d
__default_activation = nn.LeakyReLU


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


# Enhanced Shufflenetv2 for backbone
# Arxiv: https://arxiv.org/pdf/2111.00902.pdf
class ShufflenetES(nn.Module):
    # channel_ratio: [0.875, 0.5, 0.5, 0.5, 0.625, 0.5, 0.625, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    def __init__(self):
        super().__init__()
        self.stem_conv_3x3 = _conv_norm_act(3, 24, 3, 2, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage_1 = nn.Sequential(
            ESBlockS2(24, int(32 * 0.875), 32),
            InvertedResidual(32, 32 * 4),
            InvertedResidual(32, 32 * 4),
        )
        self.stage_2 = nn.Sequential(
            ESBlockS2(32, int(48 * 0.5), 48),
            InvertedResidual(48, 48 * 4),
            InvertedResidual(48, 48 * 4),
            InvertedResidual(48, 48 * 4),
            InvertedResidual(48, 48 * 4),
            InvertedResidual(48, 48 * 4),
            InvertedResidual(48, 48 * 4),
        )
        self.stage_3 = nn.Sequential(
            ESBlockS2(48, int(96 * 0.5), 96),
            InvertedResidual(96, 96 * 4),
            InvertedResidual(96, 96 * 4),
        )

    def forward(self, x):
        x = self.stem_conv_3x3(x)
        x = self.maxpool(x)
        c3 = self.stage_1(x)
        c4 = self.stage_2(c3)
        c5 = self.stage_3(c4)
        return [c3, c4, c5]


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
        c4 = torch.cat([c4, F.interpolate(c5, scale_factor=2, mode="bilinear")], dim=1)
        c4 = self.c4_csp_1(c4)
        c3 = torch.cat([c3, F.interpolate(c4, scale_factor=2, mode="bilinear")], dim=1)
        c3 = self.c3_csp(c3)
        c4 = torch.cat([c4, self.downsample_1(c3)], dim=1)
        c4 = self.c4_csp_2(c4)
        c5 = torch.cat([c5, self.downsample_2(c4)], dim=1)
        c5 = self.c5_csp(c5)
        return [c3, c4, c5]


# Decoupled Head from YOLOX
class DecoupledHead(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.squeeze = pointwise_conv(in_channels, mid_channels)
        self.branch_box = nn.Sequential(
            depthwise_conv(mid_channels, 5, 1),
            pointwise_conv(mid_channels, mid_channels),
            depthwise_conv(mid_channels, 5, 1),
            pointwise_conv(mid_channels, 4, None, None),
        )
        self.branch_cls = nn.Sequential(
            depthwise_conv(mid_channels, 3, 1),
            pointwise_conv(mid_channels, mid_channels),
            depthwise_conv(mid_channels, 3, 1),
            pointwise_conv(mid_channels, num_classes+1, None, None),
        )

    def forward(self, x):
        x = self.squeeze(x)
        x1 = self.branch_box(x)
        x2 = self.branch_cls(x)
        x = torch.cat((x1, x2), dim=1)  # bbox + (obj + cls)
        return x


# Detection Head modified from yolov5 series
# Ultralytics version
class DetectHead(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes, anchors):  # detection layer
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
        self.mode = "normal"
        self.m = nn.ModuleList(
            DecoupledHead(in_channels, mid_channels, num_classes) for _ in range(self.nl)
        )  # output conv

    @staticmethod
    def _make_grid(nx, ny):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        return x

    def inference(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
            y = x[i].sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.view(bs, -1, self.no))
        return torch.cat(z, 1), x

    def dforward(self, x):
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
        return x

    def dsp(self):
        self.forward = self.dforward
        return self


# Full detection model
class YoloxShuffleNetES(nn.Module):
    def __init__(self, num_classes, anchors):
        super().__init__()
        self.backbone = ShufflenetES()
        self.neck = CSPPAN([32, 48, 96], 48)
        self.head = DetectHead(48, 48, num_classes, anchors)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def inference(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head.inference(x)
        return x

    def dsp(self):
        self.head.dsp()
        return self


def yolox_esmk_shrink(num_classes):
    model = YoloxShuffleNetES(
        num_classes=num_classes,
        anchors=([11, 14], [30, 46], [143, 130]),
    )
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
    return model


if __name__ == "__main__":
    model = yolox_esmk_shrink(num_classes=3)
    # model.load_state_dict(torch.load("./best.pt", map_location="cpu")["state_dict"])
    for y in model(torch.rand(4, 3, 224, 416)):
        print(y.shape)
