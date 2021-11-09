import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoupledHead(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super().__init__()
        # 128 -> 128
        # (128 3x3-> 128)*2 -> 80
        # (128 3x3-> 128)*2 -> 5
        self.squeeze = convpack_1x1(in_channels, hidden_channels)
        self.branch_1 = nn.Sequential(
            convpack_3x3_1x(hidden_channels),
            convpack_1x1(hidden_channels, hidden_channels),
            convpack_3x3_1x(hidden_channels),
            convpack_1x1(hidden_channels, num_classes, 'linear', disable_norm=True),
        )
        self.branch_2 = nn.Sequential(
            convpack_3x3_1x(hidden_channels),
            convpack_1x1(hidden_channels, hidden_channels),
            convpack_3x3_1x(hidden_channels),
            convpack_1x1(hidden_channels, 5, 'linear', disable_norm=True),
        )

    def forward(self, x):
        x = self.squeeze(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x = torch.cat((x2, x1), dim=1) # bboxes + classes
        return x


class DetectHead(nn.Module):
    def __init__(self, num_classes, anchors, ch):  # detection layer
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

        # 128 -> 85 * 3
        # self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.m = nn.ModuleList(DecoupledHead(x, num_classes) for x in ch)  # output conv

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def forward(self, x):
        # x = list(x)
        # x = x.copy()  # for profiling
        # z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        return x

    def inference(self, x):
        # x = list(x)
        # x = x.copy()  # for profiling
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
            y = x[i].sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.view(bs, -1, self.no))
        return torch.cat(z, 1), x

    def dsp_forward(self, x):
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
        return x

    def dsp(self):
        self.forward = self.dsp_forward
        return self


class ConvPack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, activation, disable_norm=False):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False))
        if not disable_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation == "relu":
            layers.append(nn.ReLU())
        self.core = nn.Sequential(*layers)

    def forward(self, x):
        return self.core(x)


def convpack_1x1(in_channels, out_channels, activation="relu", disable_norm=False):
    return ConvPack(in_channels, out_channels, 1, 1, 0, 1, activation, disable_norm)


def convpack_3x3_1x(in_channels):
    return ConvPack(in_channels, in_channels, 3, 1, 1, in_channels, "relu")


def convpack_3x3_2x(in_channels):
    return ConvPack(in_channels, in_channels, 3, 2, 1, in_channels, "relu")


def convpack_3x3_2x_dense(in_channels, out_channels):
    return ConvPack(in_channels, out_channels, 3, 2, 1, 1, "relu")


class CSPBlock(nn.Module):
    def __init__(self, downsample, dense_block, reshape):
        super().__init__()
        self.downsample = downsample
        self.dense_block = dense_block
        self.reshape = reshape

    def forward(self, x):
        x1 = self.dense_block(x)
        x2 = self.downsample(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.reshape(x)
        return x


class ShortcutBlock(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, x):
        return x + self.core(x)


def reversed_bottleneck(c1, c2, c3):
    return nn.Sequential(
        convpack_1x1(c1, c2),
        convpack_3x3_1x(c2),
        convpack_1x1(c2, c3, "linear"),
    )


def residual_reversed_bottleneck(c1, c2, c3):
    return ShortcutBlock(reversed_bottleneck(c1, c2, c3))


class CSPMobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_dense = convpack_3x3_2x_dense(3, 32)
        self.block_1 = CSPBlock(
            convpack_3x3_1x(32),
            reversed_bottleneck(32, 64, 16),
            convpack_1x1(48, 48),  # 32 + 16 = 48 -> 48
        )
        self.block_2 = CSPBlock(
            convpack_3x3_2x(48),
            nn.Sequential(
                convpack_3x3_2x(48),
                convpack_1x1(48, 24, "linear"),
                residual_reversed_bottleneck(24, 96, 24),
            ),
            convpack_1x1(72, 72),  # 48 + 24 = 72 -> 72
        )
        self.block_3 = CSPBlock(
            convpack_3x3_2x(72),
            nn.Sequential(
                convpack_3x3_2x(72),
                convpack_1x1(72, 32, "linear"),
                residual_reversed_bottleneck(32, 128, 32),
                residual_reversed_bottleneck(32, 128, 32),
                reversed_bottleneck(32, 128, 64),
                residual_reversed_bottleneck(64, 256, 64),
                residual_reversed_bottleneck(64, 256, 64),
                residual_reversed_bottleneck(64, 256, 64),
            ),
            convpack_1x1(136, 192),  # 72 + 64 = 136 -> 192
        )
        self.block_4 = CSPBlock(
            convpack_3x3_2x(192),
            nn.Sequential(
                convpack_3x3_2x(192),
                convpack_1x1(192, 96, "linear"),
                residual_reversed_bottleneck(96, 384, 96),
                residual_reversed_bottleneck(96, 384, 96),
            ),
            convpack_1x1(288, 288),  # 192 + 96 = 288 -> 288
        )
        self.block_5 = CSPBlock(
            convpack_3x3_2x(288),
            nn.Sequential(
                convpack_3x3_2x(288),
                convpack_1x1(288, 160, "linear"),
                residual_reversed_bottleneck(160, 640, 160),
                residual_reversed_bottleneck(160, 640, 160),
                reversed_bottleneck(160, 640, 320),
            ),
            convpack_1x1(608, 640),  # 288+ 320 = 608 -> 640
        )

    def forward(self, x):
        x = self.conv_dense(x)
        x = self.block_1(x)
        x = self.block_2(x)
        y1 = self.block_3(x)
        y2 = self.block_4(y1)
        y3 = self.block_5(y2)
        return [y1, y2, y3]


class PathAggregationPyramid(nn.Module):
    def __init__(self, in_channels_list, out_channels_list):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        for in_channels, out_channels in zip(in_channels_list, out_channels_list):
            self.lateral_convs.append(convpack_1x1(in_channels, out_channels))

    def forward(self, x):
        laterals = [lateral_conv(x[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2, mode="bilinear")
        # build outputs
        # part 1: from original levels
        inter_outs = [laterals[i] for i in range(used_backbone_levels)]
        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            inter_outs[i + 1] += F.interpolate(inter_outs[i], scale_factor=0.5, mode="bilinear")
        outs = []
        outs.append(inter_outs[0])
        outs.extend([inter_outs[i] for i in range(1, used_backbone_levels)])
        return outs


# class TestBackbone(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.base = nn.Sequential(
#             nn.Conv2d(3, 1, 3),
#             nn.Conv2d(1, 1, 3, 2, 1),
#             nn.Conv2d(1, 1, 3, 2, 1),
#             nn.Conv2d(1, 192, 3, 2, 1),
#         )
#         self.s1 = nn.Conv2d(192, 288, 3, 2, 1)
#         self.s2 = nn.Conv2d(288, 640, 3, 2, 1)
    
#     def forward(self, x):
#         x1 = self.base(x)
#         x2 = self.s1(x1)
#         x3 = self.s2(x2)
#         return [x1, x2, x3]

class YoloCSPMobilenetV2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = CSPMobileNetV2()
        self.pan = PathAggregationPyramid([192, 288, 640], [128, 128, 128])
        self.detect = DetectHead(
            num_classes=num_classes,
            anchors=([11, 14], [30, 46], [143, 130]),
            ch=(128, 128, 128),
        )
        for m in self.modules():
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

    def forward(self, x):
        x = self.backbone(x)
        x = self.pan(x)
        x = self.detect(x)
        return x

    def inference(self, x):
        x = self.backbone(x)
        x = self.pan(x)
        x = self.detect.inference(x)
        return x

    def dsp(self):
        self.detect.dsp()
        return self


def yolox_cspm_depthwise_test(num_classes=3):
    return YoloCSPMobilenetV2(num_classes)


if __name__ == "__main__":
    model = yolox_cspm_depthwise_test()
    # model.load_state_dict(torch.load("./best.pt", map_location="cpu")["state_dict"])
    for y in model(torch.rand(4, 3, 224, 416)):
        print(y.shape)
