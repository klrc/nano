import torch.nn as nn

__default_norm = nn.BatchNorm2d
__default_activation = nn.ReLU6


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
