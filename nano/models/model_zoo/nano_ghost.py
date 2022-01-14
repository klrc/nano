import torch
import torch.nn as nn
from nano.models.backbones.enhanced_shufflenet_v2 import (
    EnhancedShuffleNetv2_3x,
    EnhancedShuffleNetv2_4x,
)
from nano.models.necks.ghost_pan import GhostPAN
from nano.models.heads.nanodet_head import NanoHead



# Params size (MB): 3.12
class GhostNano_3x4_m96(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone_width, neck_width, strides = ((24, 96, 192, 288), 96, (8, 16, 32, 64))
        self.backbone = EnhancedShuffleNetv2_3x(backbone_width)
        self.neck = GhostPAN(backbone_width[1:], neck_width, len(strides))
        self.head = NanoHead(neck_width, neck_width, strides, num_classes)
        self.num_classes = num_classes
        self.strides = strides

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


# Params size (MB): 3.12
class GhostNano_4x3_m96(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone_width, neck_width, strides = ((24, 48, 96, 192, 288), 96, (8, 16, 32))
        self.backbone = EnhancedShuffleNetv2_4x(backbone_width)
        self.neck = GhostPAN(backbone_width[1:], neck_width, len(strides))
        self.head = NanoHead(neck_width, neck_width, strides, num_classes)
        self.num_classes = num_classes
        self.strides = strides

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


# Params size (MB): 2.93
class GhostNano_3x3_m96(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone_width, neck_width, strides = ((24, 96, 192, 288), 96, (8, 16, 32))
        self.backbone = EnhancedShuffleNetv2_3x(backbone_width)
        self.neck = GhostPAN(backbone_width[1:], neck_width, len(strides))
        self.head = NanoHead(neck_width, neck_width, strides, num_classes)
        self.num_classes = num_classes
        self.strides = strides

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


# Params size (MB): 1.36
class GhostNano_3x3_s64(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone_width, neck_width, strides = ((24, 64, 128, 192), 64, (8, 16, 32))
        self.backbone = EnhancedShuffleNetv2_3x(backbone_width)
        self.neck = GhostPAN(backbone_width[1:], neck_width, len(strides))
        self.head = NanoHead(neck_width, neck_width, strides, num_classes)
        self.num_classes = num_classes
        self.strides = strides

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

# Params size (MB): 1.13
class GhostNano_3x3_s32(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone_width, neck_width, strides = ((24, 64, 128, 192), 32, (8, 16, 32))
        self.backbone = EnhancedShuffleNetv2_3x(backbone_width)
        self.neck = GhostPAN(backbone_width[1:], neck_width, len(strides))
        self.head = NanoHead(neck_width, neck_width, strides, num_classes)
        self.num_classes = num_classes
        self.strides = strides

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    from nano.models.utils import check_size

    model = GhostNano_3x3_m96(3)
    x = torch.rand(1, 3, 224, 416)
    model.eval()
    y = model(x)
    for yi in y:
        print(yi.shape)
    check_size(model)
