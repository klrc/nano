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


class AnchorfreeHead(nn.Module):
    """
    Anchor-free post process module.
    """

    def __init__(self, in_channels=96, hidden_channels=96, strides=(8, 16, 32), num_classes=80):
        super().__init__()
        self.headless = NanoHeadless(in_channels, hidden_channels, strides, num_classes)
        self.strides = strides

    def forward(self, xs):
        """
        returns:
            ys: (N, A, 2+1+4+c (grid + stride + box_xyxy + cls))
        """

        # feed forward
        xs = self.headless(xs)

        # post process forward
        ys = []
        for x, stride in zip(xs, self.strides):
            n, _, h, w = x.shape
            device = x.device
            # make grids
            yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)], indexing="ij")
            G = torch.stack((xv, yv), 2).view(1, -1, 2).to(device)
            if self.training:
                G = G.repeat(n, 1, 1)
                # make stride
                S = torch.tensor(stride).expand(n, h * w, 1).to(device)
                # flatten anchors
                y = x.flatten(2).permute(0, 2, 1)
                y = torch.cat((G, S, y), dim=-1)
                # post process
                y[..., 3 : 3 + 4] = y[..., 3 : 3 + 4].sigmoid() * 5 * stride  # sigmoid box only, max = 5 * 2 * stride
                y[..., 3 : 3 + 2] *= -1
                y[..., 3 : 3 + 4] += (G.repeat(1, 1, 2) + 0.5) * stride
                ys.append(y)
            else:
                # flatten anchors
                y = x.flatten(2).permute(0, 2, 1).sigmoid()
                # post process
                y[..., : 4] *= 5 * stride  # sigmoid box only, max = 5 * 2 * stride
                y[..., : 2] *= -1
                y[..., : 4] += (G.repeat(1, 1, 2) + 0.5) * stride
                ys.append(y)

        return torch.cat(ys, dim=1)
