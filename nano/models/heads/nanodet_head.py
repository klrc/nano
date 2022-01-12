import torch
import torch.nn as nn

from ..multiplex.conv import depthwise_conv, pointwise_conv


class NanoHead(nn.Module):
    """
    NanoDet head with QFL only (coupled)
    """

    def __init__(self, in_channels=(96, 96, 96), hidden_channels=96, strides=(8, 16, 32), num_classes=80):
        super().__init__()
        self.convs = nn.ModuleList()
        self.strides = strides
        self.debug = False
        assert len(strides) == len(in_channels)

        for i in range(len(in_channels)):
            assert in_channels[i] == hidden_channels
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

    def make_grids(self, x):
        h, w = x.shape[-2:]
        yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
        grid = torch.stack((xv, yv), 2).view(1, -1, 2).to(x.device)
        return grid

    def forward(self, xs):
        """
        returns:
            if training:
                ys: (N, A, < xyxy | obj | c >)
                gs: grid mask for outputs
                ss: stride mask for outputs
            else:
                ys: (N, A, < xyxy | obj | c >)
        """
        ys = []
        gs = []
        ss = []
        for i, (x, stride) in enumerate(zip(xs, self.strides)):
            y = self.convs[i](x)

            # record stride mask
            stride_mask = torch.tensor(stride).expand(1, y.size(-2) * y.size(-1)).to(y.device)
            grid_mask = self.make_grids(y)
            ss.append(stride_mask)
            gs.append(grid_mask)

            # output shaped as (N, A, 4+c)
            y = y.flatten(2).permute(0, 2, 1)
            if not self.training:
                y = y.sigmoid()
            else:
                y[..., :4] = y[..., :4].sigmoid()

            # ccwh -> xywh
            y[..., :2] = (y[..., :2] * 7 - 3 + grid_mask) * stride  # +-3.5
            y[..., 2:4] = torch.exp(y[..., 2:4] * 3) * stride  # max=20*stride

            # xywh -> xyxy
            y[..., :2] -= y[..., 2:4] / 2
            y[..., 2:4] += y[..., :2]
            ys.append(y)

        # concat outputs as (N, A1+A2...+Ai, 4+c)
        if self.training or self.debug:
            return torch.cat(ys, 1), torch.cat(gs, 1), torch.cat(ss, 1)
        return torch.cat(ys, 1)


class NanoHeadless(nn.Module):
    """
    headless version of NanoHead for porting
    """

    def __init__(self, head: NanoHead):
        super().__init__()
        self.convs = head.convs

    def forward(self, xs):
        ys = []
        for i, x in enumerate(xs):
            y = self.convs[i](x)
            ys.append(y)
        return ys
