import torch
import torch.nn as nn
from nano.models.backbones.enhanced_shufflenet_v2 import (
    EnhancedShuffleNetv2_x4_l,
    EnhancedShuffleNetv2_x3_m,
    EnhancedShuffleNetv2_x3_s,
    EnhancedShuffleNetv2_x4_m,
)
from nano.models.necks.ghost_pan import (
    GhostPAN_4x4,
    GhostPAN_4x3,
    GhostPAN_3x3,
)
from nano.models.heads.nanodet_head import NanoHead


# Params size (MB): 7.21
class GhostNano_4x4_l128(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = EnhancedShuffleNetv2_x4_l()
        self.neck = GhostPAN_4x4([64, 128, 256, 512], 128)
        self.head = NanoHead((128, 128, 128, 128), 128, num_classes=num_classes)
        self.num_classes = num_classes
        self.strides = (8, 16, 32)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

# Params size (MB): 3.31
class GhostNano_4x4_m96(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = EnhancedShuffleNetv2_x4_m()
        self.neck = GhostPAN_4x4([48, 96, 192, 288], 96)
        self.head = NanoHead((96, 96, 96, 96), 96, num_classes=num_classes)
        self.num_classes = num_classes
        self.strides = (8, 16, 32)

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
        self.neck = GhostPAN_4x3([48, 96, 192, 288], 96)
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
        self.neck = GhostPAN_3x3([96, 192, 288], 96)
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
        self.neck = GhostPAN_3x3([64, 128, 192], 32)
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

    model = GhostNano_3x3_s32(3)
    x = torch.rand(4, 3, 224, 416)
    model.eval()
    y = model(x)
    print(y[0][100])
    check_size(model)
