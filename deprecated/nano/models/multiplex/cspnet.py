import torch
import torch.nn as nn
from .conv import pointwise_conv
from .mobilenet import InvertedResidual

# Cross Stage Partial Network block
# fusion-first mode
# CVF: https://openaccess.thecvf.com/content_CVPRW_2020/papers/w28/Wang_CSPNet_A_New_Backbone_That_Can_Enhance_Learning_Capability_of_CVPRW_2020_paper.pdf
class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dense_block=InvertedResidual):
        super().__init__()
        self.short_conv = pointwise_conv(in_channels, in_channels // 2)
        self.main_conv = pointwise_conv(in_channels, in_channels // 2)
        self.dense_block = dense_block(in_channels // 2, in_channels, in_channels // 2)
        self.post_transition = pointwise_conv(in_channels, out_channels)

    def forward(self, x):
        x1 = self.short_conv(x)
        x2 = self.main_conv(x)
        x2 = self.dense_block(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.post_transition(x)
        return x