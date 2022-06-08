import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


__default_norm = nn.BatchNorm2d
__default_activation = nn.ReLU6

__all__ = ["VoVNet", "vovnet27_slim", "vovnet39", "vovnet57"]


model_urls = {
    "vovnet39": "https://dl.dropbox.com/s/1lnzsgnixd8gjra/vovnet39_torchvision.pth?dl=1",
    "vovnet57": "https://dl.dropbox.com/s/6bfu9gstbwfw31m/vovnet57_torchvision.pth?dl=1",
}


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


def conv3x3(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [
        (
            "{}_{}/conv".format(module_name, postfix),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
        ),
        ("{}_{}/norm".format(module_name, postfix), __default_norm(out_channels)),
        ("{}_{}/relu".format(module_name, postfix), __default_activation(inplace=True)),
    ]


def conv1x1(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=1, padding=0):
    """1x1 convolution"""
    return [
        (
            "{}_{}/conv".format(module_name, postfix),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
        ),
        ("{}_{}/norm".format(module_name, postfix), __default_norm(out_channels)),
        ("{}_{}/relu".format(module_name, postfix), __default_activation(inplace=True)),
    ]


# basic conv+norm+relu combination
def _conv_norm_act(in_channels, out_channels, kernel_size, stride, padding, groups, norm=__default_norm, act=__default_activation):
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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

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
    return _vovnet("vovnet27_slim", [32, 48, 64, 96], [64, 128, 256, 384], [1, 1, 1, 1], 5, pretrained, progress, **kwargs)


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


class VoVYOLO(nn.Module):
    def __init__(
        self,
        num_classes,
        anchors=[
            [10, 13, 16, 30, 33, 23],
            [10, 13, 16, 30, 33, 23],
            [10, 13, 16, 30, 33, 23],
        ],
        cf=None,
    ):
        super().__init__()
        outchannel = 48  # 96
        self.backbone = vovnet27_extra_slim()
        channels = (128, 256, 384)
        self.neck = CSPPANS3(channels, outchannel)  # [96, 192, 384]
        self.detect = DetectHead([outchannel, outchannel, outchannel], num_classes, anchors, strides=(8, 16, 32), cf=cf)  # head
        initialize_weights(self)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.detect(x)  # head
        return x

    def fuse(self):
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
    model = VoVYOLO(
        6,
        anchors=[
            [10, 13, 16, 30, 33, 23],
            [10, 13, 16, 30, 33, 23],
            [10, 13, 16, 30, 33, 23],
        ],
        cf=cf,
    )

    # forward test
    for y in model.forward(torch.rand(4, 3, 384, 640)):
        print(y.shape)

    # size & bandwidth test
    get_model_info(model, (1, 3, 384, 640))

    # onnx test
    model.eval().fuse()
    torch.onnx.export(model, torch.rand(1, 3, 384, 640), "vovnet_yolov5.onnx", opset_version=12)
