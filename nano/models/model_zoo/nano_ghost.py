import torch
import torch.nn as nn
from nano.models.backbones.enhanced_shufflenet_v2 import (
    EnhancedShuffleNetv2_x4_l,
    EnhancedShuffleNetv2_x3_m,
    EnhancedShuffleNetv2_x3_s,
    EnhancedShuffleNetv2_x4_m,
)
from nano.models.necks.ghost_pan import GhostPAN
from nano.models.heads.nanodet_head import NanoHead


# Params size (MB): 3.03
class GhostNano_3x4_m96(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.strides = (8, 16, 32, 64)
        self.backbone = EnhancedShuffleNetv2_x3_m()
        self.neck = GhostPAN([96, 192, 288], (96, 96, 96, 96))
        self.head = NanoHead((96, 96, 96, 96), 96, self.strides, num_classes=num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


# Params size (MB): 3.12
class GhostNano_4x3_m96(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = EnhancedShuffleNetv2_x4_m()
        self.neck = GhostPAN((48, 96, 192, 288), (96, 96, 96))
        self.head = NanoHead((96, 96, 96), 96, num_classes=num_classes)
        self.num_classes = num_classes
        self.strides = (8, 16, 32)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


# Params size (MB): 2.93
class GhostNano_3x3_m96(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = EnhancedShuffleNetv2_x3_m()
        self.neck = GhostPAN((96, 192, 288), (96, 96, 96))
        self.head = NanoHead((96, 96, 96), 96, num_classes=num_classes)
        self.num_classes = num_classes
        self.strides = (8, 16, 32)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


# Params size (MB): 1.13
class GhostNano_3x3_s32(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = EnhancedShuffleNetv2_x3_s()
        self.neck = GhostPAN((64, 128, 192), (32, 32, 32))
        self.head = NanoHead((32, 32, 32), 32, num_classes=num_classes)
        self.num_classes = num_classes
        self.strides = (8, 16, 32)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    from nano.models.utils import check_size

    model = GhostNano_3x4_m96(3)
    x = torch.rand(4, 3, 224, 416)
    model.eval()
    y = model(x)
    print(y[0][100])
    check_size(model)
