import math
import warnings
import torch.nn as nn
import torch

ACTIVATION = nn.ReLU6


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else (x // 2 for x in k)  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = ACTIVATION() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class LayerBuilder:
    def __init__(self, init_channels, width_multiple, depth_multiple):
        self.width_multiple = width_multiple
        self.depth_multiple = depth_multiple
        self.last_layer_channels = init_channels

    def double_channels(self):
        self.last_layer_channels *= 2

    def build(self, module, out_channels, *args):
        out_channels = int(out_channels * self.width_multiple)
        if issubclass(module, C3):
            n = int(math.ceil(args[0] * self.depth_multiple))
            layer = C3(self.last_layer_channels, out_channels, n, *args[1:])
        else:
            layer = module(self.last_layer_channels, out_channels, *args)
        self.last_layer_channels = out_channels
        return layer


# Implementation ----------------------------------------------------------------
# # Parameters
# nc: 80  # number of classes
# depth_multiple: 0.33  # model depth multiple
# width_multiple: 0.50  # layer channel multiple

# # YOLOv5 v6.0 backbone
# backbone:
#   # [from, number, module, args]
#   [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
#    [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
#    [-1, 3, C3, [128]],
#    [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
#    [-1, 6, C3, [256]],
#    [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
#    [-1, 9, C3, [512]],
#    [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
#    [-1, 3, C3, [1024]],
#    [-1, 1, SPPF, [1024, 5]],  # 9
#   ]
# neck:
#   [[-1, 1, Conv, [512, 1, 1]],
#    [-1, 1, nn.Upsample, [None, 2, 'nearest']],
#    [[-1, 6], 1, Concat, [1]],  # cat backbone P4
#    [-1, 3, C3, [512, False]],  # 13

#    [-1, 1, Conv, [256, 1, 1]],
#    [-1, 1, nn.Upsample, [None, 2, 'nearest']],
#    [[-1, 4], 1, Concat, [1]],  # cat backbone P3
#    [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

#    [-1, 1, Conv, [256, 3, 2]],
#    [[-1, 14], 1, Concat, [1]],  # cat head P4
#    [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

#    [-1, 1, Conv, [512, 3, 2]],
#    [[-1, 10], 1, Concat, [1]],  # cat head P5
#    [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)
# head:
#    [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
#   ]
class BBv5(nn.Module):
    def __init__(self, width_multiple=0.25, depth_multiple=0.33):
        super().__init__()
        cc = LayerBuilder(3, width_multiple, depth_multiple)
        self.l4 = nn.Sequential(
            cc.build(Conv, 64, 6, 2, 2),  # 0-P1/2
            cc.build(Conv, 128, 3, 2),  # 1-P2/4
            cc.build(C3, 128, 3),
            cc.build(Conv, 256, 3, 2),  # 3-P3/8
            cc.build(C3, 256, 6),
        )
        self.l6 = nn.Sequential(
            cc.build(Conv, 512, 3, 2),  # 5-P4/16
            cc.build(C3, 512, 9),
        )
        self.l10 = nn.Sequential(
            cc.build(Conv, 1024, 3, 2),  # 7-P5/32
            cc.build(C3, 1024, 3),
            cc.build(SPPF, 1024, 5),  # 9
            cc.build(Conv, 512, 1, 1),
        )
        # 4, 6, 10, 14, 17, 20, 23
        self.l11 = nn.Upsample(None, 2, "bilinear", align_corners=True)
        cc.double_channels()  # self.l12 = <Concat>  # cat backbone P4
        self.l14 = nn.Sequential(
            cc.build(C3, 512, 3, False),
            cc.build(Conv, 256, 1, 1),
        )
        self.l15 = nn.Upsample(None, 2, "bilinear", align_corners=True)
        cc.double_channels()  # self.l16 = <Concat>  # cat backbone P3
        self.l17 = cc.build(C3, 256, 3, False)  # 17 (P3/8-small)
        self.l18 = cc.build(Conv, 256, 3, 2)
        cc.double_channels()  # self.l19 = <Concat>  # cat head P4
        self.l20 = cc.build(C3, 512, 3, False)  # 20 (P4/16-medium)
        self.l21 = cc.build(Conv, 512, 3, 2)
        cc.double_channels()  # self.l22 = <Concat>  # cat head P5
        self.l23 = cc.build(C3, 1024, 3, False)  # 23 (P5/32-large)
        self.out_channels = [int(x * width_multiple) for x in (256, 512, 1024)]
        self.strides = (8, 16, 32)

    def forward(self, x):
        backbone_x4 = self.l4(x)
        backbone_x6 = self.l6(backbone_x4)
        backbone_x10 = self.l10(backbone_x6)
        neck_x14 = self.l11(backbone_x10)
        neck_x14 = torch.cat((neck_x14, backbone_x6), dim=1)
        neck_x14 = self.l14(neck_x14)
        neck_x17 = self.l15(neck_x14)
        neck_x17 = torch.cat((neck_x17, backbone_x4), dim=1)
        neck_x17 = self.l17(neck_x17)
        neck_x20 = self.l18(neck_x17)
        neck_x20 = torch.cat((neck_x20, neck_x14), dim=1)
        neck_x20 = self.l20(neck_x20)
        neck_x23 = self.l21(neck_x20)
        neck_x23 = torch.cat((neck_x23, backbone_x10), dim=1)
        neck_x23 = self.l23(neck_x23)
        return [neck_x17, neck_x20, neck_x23]
