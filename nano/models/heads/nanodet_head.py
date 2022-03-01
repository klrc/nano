import torch
import torch.nn as nn

from ..multiplex.conv import depthwise_conv, pointwise_conv


class NanoHeadless(nn.Module):
    """
    headless version of NanoHead
    """

    def __init__(self, in_channels=96, hidden_channels=96, strides=(8, 16, 32), num_classes=80):
        super().__init__()
        self.convs = nn.ModuleList()
        assert in_channels == hidden_channels
        for _ in range(len(strides)):
            out_channels = 4 + num_classes
            self.convs.append(
                nn.Sequential(
                    depthwise_conv(int(hidden_channels), 5, 1),
                    pointwise_conv(int(hidden_channels), int(hidden_channels)),
                    depthwise_conv(int(hidden_channels), 5, 1),
                    pointwise_conv(int(hidden_channels), int(hidden_channels)),
                    pointwise_conv(int(hidden_channels), int(out_channels), norm=None, act=None),
                )
            )

    def forward(self, xs):
        ys = []
        for i, x in enumerate(xs):
            y = self.convs[i](x)
            ys.append(y)
        return ys


class Anchorfree(nn.Module):
    """
    Anchor-free post process module.
    """

    def __init__(self, strides=(8, 16, 32)):
        super().__init__()
        self.strides = strides

    def make_grids(self, ys):
        ret = []
        for y in ys:
            h, w = y.shape[-2:]
            yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)], indexing="ij")
            ret.append(torch.stack((xv, yv), 2).view(1, -1, 2).to(y.device))
        return ret

    def make_strides(self, ys):
        ret = []
        for y, stride in zip(ys, self.strides):
            ret.append(torch.tensor(stride).expand(1, y.size(-2) * y.size(-1)).to(y.device))
        return ret

    def forward(self, ys):
        """
        gs: grid mask for outputs
        ss: stride mask for outputs
        returns:
            ys: (N, A, < xyxy | obj | c >)
        """
        gm = self.make_grids(ys)
        sm = self.make_strides(ys)

        ret = []
        for i, y in enumerate(ys):
            # output shaped as (N, A, 4+c)
            y = y.flatten(2).permute(0, 2, 1)
            if not self.training:
                y = y.sigmoid()
            else:
                y[..., :4] = y[..., :4].sigmoid()

            # ccwh -> xywh
            y[..., :2] = (y[..., :2] * 7 - 3 + gm[i]) * self.strides[i]  # +-3.5
            y[..., 2:4] = torch.exp(y[..., 2:4] * 3) * self.strides[i]  # max=20*stride

            # xywh -> xyxy
            y[..., :2] -= y[..., 2:4] / 2
            y[..., 2:4] += y[..., :2]
            ret.append(y)

        # concat outputs as (N, A1+A2...+Ai, 4+c)
        return torch.cat(ret, 1), torch.cat(gm, 1), torch.cat(sm, 1)


class NanoHead(nn.Module):
    def __init__(self, in_channels=96, hidden_channels=96, strides=(8, 16, 32), num_classes=80) -> None:
        super().__init__()
        self.headless = NanoHeadless(in_channels, hidden_channels, strides, num_classes)
        self.postprocess = Anchorfree(strides)

    def forward(self, x):
        x = self.headless(x)
        x = self.postprocess(x)
        return x
