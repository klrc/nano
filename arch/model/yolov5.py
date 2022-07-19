import torch.nn as nn
import torch

from .detection_utils.backbone_v5 import Conv, DWConv, ACTIVATION
from .detection_utils import BBv5, DH41C


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


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


class Yolov5(nn.Module):
    def __init__(self, num_classes=80, width_multiple=0.25, depth_multiple=0.33) -> None:
        super().__init__()
        self.backbone = BBv5(width_multiple, depth_multiple)
        self.head = DH41C(self.backbone.out_channels, num_classes, strides=self.backbone.strides)
        self.initialize_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def replace_activation(self, act):
        for m in self.modules():
            for cname, c in m.named_children():
                if isinstance(c, ACTIVATION):
                    setattr(m, cname, act())
        return self

    def initialize_params(self):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True
        return self

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        for m in self.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.head.post_process = False
        return self

    def summary(self, tsize=(4, 3, 416, 416)):
        from loguru import logger
        from thop import profile

        img = torch.rand(tsize, device=next(self.parameters()).device)
        flops, params = profile(self, inputs=(img,), verbose=False)
        params /= 1e6
        flops /= 1e9
        logger.success(f"Yolov5 Summary @Resolution = {tsize[2:]}")
        logger.success(f"Estimated Quantsize: {params:.2f}M")
        logger.success(f"Estimated Bandwidth: {flops * 2:.2f}G")

    def load_weight(self, weight, pretrained=False):
        if isinstance(weight, str):
            map_location = None
            if not torch.cuda.is_available():
                map_location = "cpu"
            print(f"loading weight from {weight}")  # report
            sd = torch.load(weight, map_location=map_location)
        else:
            sd = weight
        # extract_from_ultralytics
        if pretrained:
            name_dict = {
                "model.0.": "backbone.l4.0.",
                "model.1.": "backbone.l4.1.",
                "model.2.": "backbone.l4.2.",
                "model.3.": "backbone.l4.3.",
                "model.4.": "backbone.l4.4.",
                "model.5.": "backbone.l6.0.",
                "model.6.": "backbone.l6.1.",
                "model.7.": "backbone.l10.0.",
                "model.8.": "backbone.l10.1.",
                "model.9.": "backbone.l10.2.",
                "model.10.": "backbone.l10.3.",
                "model.13.": "backbone.l14.0.",
                "model.14.": "backbone.l14.1.",
                "model.17.": "backbone.l17.",
                "model.18.": "backbone.l18.",
                "model.20.": "backbone.l20.",
                "model.21.": "backbone.l21.",
                "model.23.": "backbone.l23.",
                "model.24.": "head.",
            }
            csd = {}
            for k, v in sd.items():
                if pretrained:
                    for pattern in name_dict:
                        if pattern in k:
                            k = k.replace(pattern, name_dict[pattern])
                            break
                    else:
                        print(k)
                csd[k] = v
        else:
            csd = sd
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load
        print(f"transferred {len(csd)}/{len(self.state_dict())} items")  # report
        return self


def yolov5n(num_classes=80, weight=None, pretrained=False):
    model = Yolov5(num_classes, 0.25, 0.33)
    if weight:
        model.load_weight(weight, pretrained=pretrained)
    elif pretrained is True:
        raise Exception("support local state_dict file only, pls specify weight file path")
    return model


def yolov5s(num_classes=80, weight=None, pretrained=False):
    model = Yolov5(num_classes, 0.50, 0.33)
    if weight:
        model.load_weight(weight, pretrained=pretrained)
    elif pretrained is True:
        raise Exception("support local state_dict file only, pls specify weight file path")
    return model


def yolov5m(num_classes=80, weight=None, pretrained=False):
    model = Yolov5(num_classes, 0.75, 0.67)
    if weight:
        model.load_weight(weight, pretrained=pretrained)
    elif pretrained is True:
        raise Exception("support local state_dict file only, pls specify weight file path")
    return model
