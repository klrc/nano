import warnings
import torch
import torch.nn as nn
import math

WIDTH_MULTIPLE = 0.5
x64 = int(64 * WIDTH_MULTIPLE)
x128 = int(128 * WIDTH_MULTIPLE)
x256 = int(256 * WIDTH_MULTIPLE)
x512 = int(512 * WIDTH_MULTIPLE)
x1024 = int(1024 * WIDTH_MULTIPLE)

ACTIVATION = nn.ReLU6


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else (x // 2 for x in k)  # auto-pad
    return p


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


def fuse_conv_and_bn(conv, bn):
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = (
        nn.Conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, groups=conv.groups, bias=True)
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


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


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1), DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(), GhostConv(c_, c2, 1, 1, act=False)
        )  # pw  # dw  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


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


# Detection Head modified from yolov5 series
# Ultralytics version
class DetectHead(nn.Module):
    def __init__(self, in_channels, num_classes, anchors, strides, cf=None):  # detection layer=(8, 16, 32)
        super().__init__()
        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.stride = torch.tensor(strides)  # strides computed during build
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)  # shape(nl,na,2)
        self.register_buffer("anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.mode_dsp_off = True
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in in_channels)  # output conv
        self._initialize_biases(cf)

    def _initialize_biases(self, cf):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

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
        if self.training or not self.mode_dsp_off:
            return x
        else:
            return torch.cat(z, 1), x


class YoloV5SGhost(nn.Module):
    def __init__(
        self,
        num_classes,
        anchors=[
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326],
        ],
        cf=None,
    ) -> None:
        super().__init__()
        self.p1 = Conv(3, x64, 6, 2, 2)  # 0-P1/2
        self.p2 = Conv(x64, x128, 3, 2)  # 1-P2/4
        self.p3 = nn.Sequential(
            C3Ghost(x128, x128),
            Conv(x128, x256, 3, 2),  # 3-P3/8
        )
        self.p4 = nn.Sequential(
            C3Ghost(x256, x256),
            C3Ghost(x256, x256),
            Conv(x256, x512, 3, 2),  # 5-P4/16
        )
        self.p5 = nn.Sequential(
            C3Ghost(x512, x512),
            C3Ghost(x512, x512),
            C3Ghost(x512, x512),
            Conv(x512, x1024, 3, 2),  # 7-P5/32
        )
        self.sppf = nn.Sequential(
            C3Ghost(x1024, x1024),
            SPPF(x1024, x1024, 5),  # 9
            Conv(x1024, x512, 1, 1),
        )
        self.sppf_upsample = nn.Upsample(None, 2, "bilinear", align_corners=True)
        self.neck_pix = nn.Sequential(
            C3Ghost(x1024, x512, False),
            Conv(x512, x256, 1, 1),
        )
        self.neck_pix_upsample = nn.Upsample(None, 2, "bilinear", align_corners=True)
        self.neck_small = C3Ghost(x512, x256, False)
        self.neck_small_conv = Conv(x256, x256, 3, 2)
        self.neck_medium = C3Ghost(x512, x512, False)
        self.neck_medium_conv = Conv(x512, x512, 3, 2)
        self.neck_large = C3Ghost(x1024, x1024, False)
        self.detect = DetectHead([x256, x512, x1024], num_classes, anchors, strides=(8, 16, 32), cf=cf)  # head
        initialize_weights(self)

    def forward(self, x):
        x = self.p1(x)
        x = self.p2(x)
        b_p3 = self.p3(x)  # 256x
        b_p4 = self.p4(b_p3)  # 512x
        b_p5 = self.p5(b_p4)  # 1024x
        h_p5 = self.sppf(b_p5)  # 512x
        x = self.sppf_upsample(h_p5)
        x = torch.cat((x, b_p4), dim=1)  # cat backbone P4  # 1024x
        h_p4 = self.neck_pix(x)  # 256x
        x = self.neck_pix_upsample(h_p4)
        x = torch.cat((x, b_p3), dim=1)  # cat backbone P3  # 512x
        n_fs = self.neck_small(x)  # 17 (P3/8-small) # 256x
        x = self.neck_small_conv(n_fs)
        x = torch.cat((x, h_p4), dim=1)  # cat head P4  # 512x
        n_fm = self.neck_medium(x)  # 20 (P4/16-medium) # 512x
        x = self.neck_medium_conv(n_fm)
        x = torch.cat((x, h_p5), dim=1)  # cat head P5  # 1024x
        n_fl = self.neck_large(x)  # 23 (P5/32-large) # 1024x
        return self.detect([n_fs, n_fm, n_fl])  # Detect(P3, P4, P5)

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        for m in self.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.detect.mode_dsp_off = False
        return self


if __name__ == "__main__":
    from loguru import logger
    from thop import profile
    from copy import deepcopy

    def get_model_info(model, tsize):
        img = torch.rand(tsize, device=next(model.parameters()).device)
        flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
        params /= 1e6
        flops /= 1e9
        logger.success(f"Estimated Size:{params:.2f}M, Estimated Bandwidth: {flops * 2:.2f}G, Resolution: {tsize[2:]}")

    cf = torch.Tensor((0.5, 0.1, 0.1, 0.2, 0.05, 0.05))
    model = YoloV5SGhost(
        6,
        cf=cf,
    )

    # forward test
    for y in model.forward(torch.rand(4, 3, 384, 640)):
        print(y.shape)

    # size & bandwidth test
    get_model_info(model, (1, 3, 384, 640))

    # onnx test
    model.eval().fuse()
    torch.onnx.export(model, torch.rand(1, 3, 384, 640), "yolov5n.onnx", opset_version=12)