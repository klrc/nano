import torch
import torch.nn as nn

from ..multiplex.conv import depthwise_conv, pointwise_conv


class YoloXHead(nn.Module):
    def __init__(self, in_channels=(96, 96, 96), hidden_channels=96, strides=(8, 16, 32), num_classes=80):
        super().__init__()
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.strides = strides
        self.debug = False

        for i in range(len(in_channels)):
            self.stems.append(
                pointwise_conv(
                    in_channels=int(in_channels[i]),
                    out_channels=int(hidden_channels),
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    depthwise_conv(int(hidden_channels), 3, 1),
                    pointwise_conv(int(hidden_channels), int(hidden_channels)),
                    depthwise_conv(int(hidden_channels), 3, 1),
                    pointwise_conv(int(hidden_channels), int(hidden_channels)),
                    pointwise_conv(int(hidden_channels), int(num_classes), norm=None, act=None),
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    depthwise_conv(int(hidden_channels), 3, 1),
                    pointwise_conv(int(hidden_channels), int(hidden_channels)),
                    depthwise_conv(int(hidden_channels), 3, 1),
                    pointwise_conv(int(hidden_channels), int(hidden_channels)),
                    pointwise_conv(int(hidden_channels), 5, norm=None, act=None),
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
            x = self.stems[i](x)
            reg_preds = self.reg_convs[i](x)
            cls_preds = self.cls_convs[i](x)
            y = torch.cat([reg_preds, cls_preds], dim=1)

            # record stride mask
            stride_mask = torch.tensor(stride).expand(1, y.size(-2) * y.size(-1)).to(y.device)
            grid_mask = self.make_grids(y)
            ss.append(stride_mask)
            gs.append(grid_mask)

            # output shaped as (N, A, 4+1+c)
            y = y.flatten(2).permute(0, 2, 1)
            if not self.training:
                y = y.sigmoid()
            else:
                y[:4] = y[:4].sigmoid()
            # ccwh -> xywh
            y[..., :2] = (y[..., :2] * 7 - 3 + grid_mask) * stride  # +-3.5
            y[..., 2:4] = torch.exp(y[..., 2:4] * 3) * stride  # max=20*stride
            # xywh -> xyxy
            y[..., :2] -= y[..., 2:4] / 2
            y[..., 2:4] += y[..., :2]
            ys.append(y)

        # concat outputs as (N, A1+A2...+Ai, 4+1+c)
        if self.training or self.debug:
            return torch.cat(ys, 1), torch.cat(gs, 1), torch.cat(ss, 1)
        return torch.cat(ys, 1)
