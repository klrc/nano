import warnings
import torch
import torch.nn as nn
import math


ACTIVATION = nn.ReLU6
DEFAULT_ANCHORS = (
    (10, 13, 16, 30, 33, 23),
    (30, 61, 62, 45, 59, 119),
    (116, 90, 156, 198, 373, 326),
)


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else (x // 2 for x in k)  # auto-pad
    return p


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


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
        if anchors is None:
            anchors = DEFAULT_ANCHORS
        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.stride = torch.tensor(strides)  # strides computed during build
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.mode_dsp = False
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

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            if not self.mode_dsp:
                # reshape
                bs, _, ny, nx = x[i].shape
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                if not self.training:
                    # make grid
                    if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
                        self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                        self.grid[i] = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float().to(x[i].device)
                    # normalize and get xywh
                    y = x[i].sigmoid()
                    y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    z.append(y.view(bs, -1, self.no))
        if self.training or self.mode_dsp:
            return x
        else:
            return torch.cat(z, 1), x


class Yolov5U(nn.Module):
    def __init__(
        self,
        num_classes,
        anchors=None,
        width_multiple=0.25,
        depth_multiple=0.33,
        channels_per_chunk=16,
        cf=None,
    ) -> None:
        super().__init__()

        # scaled channels
        _cpc = channels_per_chunk
        x64, x128, x256, x512, x1024 = [int((x * width_multiple) // _cpc) * _cpc for x in (64, 128, 256, 512, 1024)]
        d3, d6, d9 = [int(math.ceil(x * depth_multiple)) for x in (3, 6, 9)]

        self.p0 = nn.Sequential(
            Conv(3, x64, 6, 2, 2),  # 0-P1/2
            Conv(x64, x128, 3, 2),  # 1-P2/4
            C3(x128, x128, d3),
            Conv(x128, x256, 3, 2),  # 3-P3/8
            C3(x256, x256, d6),
        )
        self.p1 = nn.Sequential(
            Conv(x256, x512, 3, 2),  # 5-P4/16
            C3(x512, x512, d9),
        )
        self.p2 = nn.Sequential(
            Conv(x512, x1024, 3, 2),  # 7-P5/32
            C3(x1024, x1024, d3),
            SPPF(x1024, x1024, 5),  # 9
            Conv(x1024, x512, 1, 1),
        )
        self.u0 = nn.Upsample(None, 2, "bilinear", align_corners=True)
        self.neck_0 = nn.Sequential(
            C3(x1024, x512, d3, shortcut=False),
            Conv(x512, x256, 1, 1),
        )
        self.u1 = nn.Upsample(None, 2, "bilinear", align_corners=True)
        self.neck_1 = C3(x512, x256, d3, shortcut=False)
        self.neck_1_conv = Conv(x256, x256, 3, 2)
        self.neck_2 = C3(x512, x512, d3, shortcut=False)
        self.neck_2_conv = Conv(x512, x512, 3, 2)
        self.neck_3 = C3(x1024, x1024, d3, shortcut=False)
        self.detect = DetectHead([x256, x512, x1024], num_classes, anchors, strides=(8, 16, 32), cf=cf)  # head
        initialize_weights(self)

    def forward(self, x):
        backbone_p3 = self.p0(x)
        backbone_p4 = self.p1(backbone_p3)
        backbone_p5 = self.p2(backbone_p4)
        x = self.u0(backbone_p5)
        x = torch.cat((x, backbone_p4), dim=1)  # cat backbone P4  # 1024x
        neck_0 = self.neck_0(x)
        x = self.u1(neck_0)
        x = torch.cat((x, backbone_p3), dim=1)  # cat backbone P3  # 512x
        neck_1 = self.neck_1(x)
        x = self.neck_1_conv(neck_1)
        x = torch.cat((x, neck_0), dim=1)  # cat head P4  # 512x
        neck_2 = self.neck_2(x)
        x = self.neck_2_conv(neck_2)
        x = torch.cat((x, backbone_p5), dim=1)  # cat head P5  # 1024x
        neck_3 = self.neck_3(x)
        return self.detect([neck_1, neck_2, neck_3])  # Detect(P3, P4, P5)

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        for m in self.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.detect.mode_dsp = True
        return self


def forced_load(model, weights):
    weight_mapper = {
        "model.0.": "p0.0.",
        "model.1.": "p0.1.",
        "model.2.": "p0.2.",
        "model.3.": "p0.3.",
        "model.4.": "p0.4.",
        "model.5.": "p1.0.",
        "model.6.": "p1.1.",
        "model.7.": "p2.0.",
        "model.8.": "p2.1.",
        "model.9.": "p2.2.",
        "model.10.": "p2.3.",
        "model.13.": "neck_0.0.",
        "model.14.": "neck_0.1.",
        "model.17.": "neck_1.",
        "model.18.": "neck_1_conv.",
        "model.20.": "neck_2.",
        "model.21.": "neck_2_conv.",
        "model.23.": "neck_3.",
        "model.24.": "detect.",
    }
    csd = {}
    map_location = None
    if not torch.cuda.is_available():
        map_location = 'cpu'
    for k, v in torch.load(weights, map_location=map_location).items():
        for pattern in weight_mapper:
            if pattern in k:
                k = k.replace(pattern, weight_mapper[pattern])
                break
        csd[k] = v
    csd = intersect_dicts(csd, model.state_dict())  # intersect
    model.load_state_dict(csd, strict=False)  # load
    print(f"transferred {len(csd)}/{len(model.state_dict())} items from {weights}")  # report
    return model


def yolov5s(num_classes, anchors=None, cf=None, weights=None, activation=None):
    model = Yolov5U(num_classes, anchors, width_multiple=0.5, depth_multiple=0.33, cf=cf)
    if activation is not None:
        for m in model.modules():
            for cname, c in m.named_children():
                if isinstance(c, ACTIVATION):
                    setattr(m, cname, activation())
    if weights is not None:
        model = forced_load(model, weights)        
    return model


def yolov5n(num_classes, anchors=None, cf=None, weights=None, activation=None):
    model = Yolov5U(num_classes, anchors, width_multiple=0.25, depth_multiple=0.33, cf=cf)
    if activation is not None:
        for m in model.modules():
            for cname, c in m.named_children():
                if isinstance(c, ACTIVATION):
                    setattr(m, cname, activation())
    if weights is not None:
        model = forced_load(model, weights)
    return model


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

    # cf = torch.Tensor((0.5, 0.1, 0.1, 0.2, 0.05, 0.05))
    model = yolov5n(80, weights=".yolov5_checkpoints/yolov5n_sd.pt")

    # forward test
    for y in model.forward(torch.rand(4, 3, 640, 640)):
        print(y.shape)

    # size & bandwidth test
    get_model_info(model, (1, 3, 640, 640))

    # # onnx test
    # model.eval().fuse()
    # torch.onnx.export(model, torch.rand(1, 3, 640, 640), "test.onnx", opset_version=12)
