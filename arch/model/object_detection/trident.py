# Implementation of Scale-Aware Trident Networks for Object Detection
# github source: https://github.com/dl19940602/TridentNet-block-pytorch/blob/master/trident.py
# paper source: https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Scale-Aware_Trident_Networks_for_Object_Detection_ICCV_2019_paper.pdf
# discussion: https://blog.csdn.net/qq_14845119/article/details/86716675
# basically, this is equivalent to within-network scale augmentation.

# NOTE: this needs a specialized TCU to train because there would be multiple loss backwards

import torch
import torch.nn as nn
import torch.nn.functional as F


class TridentConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilations=(1, 2, 3), groups=1, inf_dilation=2) -> None:
        super().__init__()
        self.weight = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groups).weight.data
        self.stride = stride
        self.dilations = dilations
        self.groups = groups
        self.inf_dilation = inf_dilation

    def forward(self, x):
        if self.training:
            ys = []
            for d in self.dilations:
                y = F.conv2d(x, self.weight, None, self.stride, d, d, self.groups)
                ys.append(y)
            return ys
        else:
            d = self.inf_dilation
            return F.conv2d(x, self.weight, None, self.stride, d, d, self.groups)


if __name__ == "__main__":
    model = TridentConv(3, 16, kernel_size=3).train()
    x = torch.rand(4, 3, 224, 416)
    y = model(x)
    for yi in y:
        print("train:", yi.shape)
    model.eval()
    y = model(x)
    print("eval:", y.shape)
