import warnings
import torch
import torch.nn as nn
import math
from collections import OrderedDict


ACTIVATION = nn.ReLU6
DEFAULT_ANCHORS = (
    (10, 13, 16, 30, 33, 23),
    (30, 61, 62, 45, 59, 119),
    (116, 90, 156, 198, 373, 326),
)


__all__ = ["VoVNet", "vovnet27_slim", "vovnet39", "vovnet57"]


model_urls = {
    "vovnet39": "https://dl.dropbox.com/s/1lnzsgnixd8gjra/vovnet39_torchvision.pth?dl=1",
    "vovnet57": "https://dl.dropbox.com/s/6bfu9gstbwfw31m/vovnet57_torchvision.pth?dl=1",
}


def conv3x3(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [
        (
            "{}_{}/conv".format(module_name, postfix),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
        ),
        ("{}_{}/norm".format(module_name, postfix), nn.BatchNorm2d(out_channels)),
        ("{}_{}/relu".format(module_name, postfix), ACTIVATION(inplace=True)),
    ]


def conv1x1(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=1, padding=0):
    """1x1 convolution"""
    return [
        (
            "{}_{}/conv".format(module_name, postfix),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
        ),
        ("{}_{}/norm".format(module_name, postfix), nn.BatchNorm2d(out_channels)),
        ("{}_{}/relu".format(module_name, postfix), ACTIVATION(inplace=True)),
    ]


class _OSA_module(nn.Module):
    def __init__(self, in_ch, stage_ch, concat_ch, layer_per_block, module_name, identity=False):
        super(_OSA_module, self).__init__()

        self.identity = identity
        self.layers = nn.ModuleList()
        in_channel = in_ch
        for i in range(layer_per_block):
            self.layers.append(nn.Sequential(OrderedDict(conv3x3(in_channel, stage_ch, module_name, i))))
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(OrderedDict(conv1x1(in_channel, concat_ch, module_name, "concat")))

    def forward(self, x):
        identity_feat = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)

        if self.identity:
            xt = xt + identity_feat

        return xt


class _OSA_stage(nn.Sequential):
    def __init__(self, in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num):
        super(_OSA_stage, self).__init__()

        if not stage_num == 2:
            self.add_module("Pooling", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        module_name = f"OSA{stage_num}_1"
        self.add_module(module_name, _OSA_module(in_ch, stage_ch, concat_ch, layer_per_block, module_name))
        for i in range(block_per_stage - 1):
            module_name = f"OSA{stage_num}_{i+2}"
            self.add_module(module_name, _OSA_module(concat_ch, stage_ch, concat_ch, layer_per_block, module_name, identity=True))


class VoVNet(nn.Module):
    def __init__(self, config_stage_ch, config_concat_ch, block_per_stage, layer_per_block, num_classes=1000):
        super(VoVNet, self).__init__()

        # Stem module
        stem = conv3x3(3, 64, "stem", "1", 2)
        stem += conv3x3(64, 64, "stem", "2", 1)
        stem += conv3x3(64, 128, "stem", "3", 2)
        self.add_module("stem", nn.Sequential(OrderedDict(stem)))

        stem_out_ch = [128]
        in_ch_list = stem_out_ch + config_concat_ch[:-1]
        self.stage_names = []
        for i in range(4):  # num_stages
            name = "stage%d" % (i + 2)
            self.stage_names.append(name)
            self.add_module(name, _OSA_stage(in_ch_list[i], config_stage_ch[i], config_concat_ch[i], block_per_stage[i], layer_per_block, i + 2))

        # self.classifier = nn.Linear(config_concat_ch[-1], num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        features = []
        for name in self.stage_names:
            x = getattr(self, name)(x)
            features.append(x)
        return features[-3:]
        # x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        # x = self.classifier(x)
        # return x


def _vovnet(arch, config_stage_ch, config_concat_ch, block_per_stage, layer_per_block, pretrained, progress, **kwargs):
    model = VoVNet(config_stage_ch, config_concat_ch, block_per_stage, layer_per_block, **kwargs)
    if pretrained:
        raise NotImplementedError
        # state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        # model.load_state_dict(state_dict)
    return model


def vovnet57(pretrained=False, progress=True, **kwargs):
    r"""Constructs a VoVNet-57 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vovnet("vovnet57", [128, 160, 192, 224], [256, 512, 768, 1024], [1, 1, 4, 3], 5, pretrained, progress, **kwargs)


def vovnet39(pretrained=False, progress=True, **kwargs):
    r"""Constructs a VoVNet-39 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vovnet("vovnet39", [128, 160, 192, 224], [256, 512, 768, 1024], [1, 1, 2, 2], 5, pretrained, progress, **kwargs)


def vovnet27_slim(pretrained=False, progress=True, **kwargs):
    r"""Constructs a VoVNet-39 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vovnet("vovnet27_slim", [64, 80, 96, 112], [128, 256, 384, 512], [1, 1, 1, 1], 5, pretrained, progress, **kwargs)


def vovnet27_extra_slim(pretrained=False, progress=True, **kwargs):
    r"""Constructs a VoVNet-39 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vovnet("vovnet27_slim", [16, 32, 64, 96], [32, 64, 128, 256], [1, 1, 1, 1], 5, pretrained, progress, **kwargs)


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
            if self.mode_dsp_off:
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
        if self.training or not self.mode_dsp_off:
            return x
        else:
            return torch.cat(z, 1), x


class Yolov5UV(nn.Module):
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
        x256, x512, x1024 = [int((x * width_multiple) // _cpc) * _cpc for x in (256, 512, 1024)]
        d3, d6, d9 = [int(x * depth_multiple) for x in (3, 6, 9)]

        self.backbone = vovnet27_extra_slim()
        self.sppf = nn.Sequential(
            C3(x1024, x1024, d3),
            SPPF(x1024, x1024, 5),  # 9
            Conv(x1024, x512, 1, 1),
        )
        self.sppf_upsample = nn.Upsample(None, 2, "bilinear", align_corners=True)
        self.neck_pix = nn.Sequential(
            C3(x1024, x512, d3, shortcut=False),
            Conv(x512, x256, 1, 1),
        )
        self.neck_pix_upsample = nn.Upsample(None, 2, "bilinear", align_corners=True)
        self.neck_small = C3(x512, x256, d3, shortcut=False)
        self.neck_small_conv = Conv(x256, x256, 3, 2)
        self.neck_medium = C3(x512, x512, d3, shortcut=False)
        self.neck_medium_conv = Conv(x512, x512, 3, 2)
        self.neck_large = C3(x1024, x1024, d3, shortcut=False)

        self.detect = DetectHead([x256, x512, x1024], num_classes, anchors, strides=(8, 16, 32), cf=cf)  # head
        initialize_weights(self)

    def forward(self, x):
        b_p3, b_p4, b_p5 = self.backbone(x)  # 128, 256, 512x
        print(b_p3.shape, b_p4.shape, b_p5.shape)
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


def yolov5_vovnet(num_classes, anchors=None, cf=None, weights=None):
    model = Yolov5UV(num_classes, anchors, width_multiple=0.25, depth_multiple=0.33, cf=cf)
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

    cf = torch.Tensor((0.5, 0.1, 0.1, 0.2, 0.05, 0.05))
    model = yolov5_vovnet(6, cf=cf)

    # forward test
    for y in model.forward(torch.rand(4, 3, 640, 640)):
        print(y.shape)

    # size & bandwidth test
    get_model_info(model, (1, 3, 640, 640))

    # onnx test
    model.eval().fuse()
    torch.onnx.export(model, torch.rand(1, 3, 640, 640), "ultralytics_yolov5_vovnet.onnx", opset_version=12)
