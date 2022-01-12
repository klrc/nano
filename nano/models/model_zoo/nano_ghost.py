import torch
import torch.nn as nn
from nano.models.backbones.enhanced_shufflenet_v2 import (
    EnhancedShuffleNetv2_3x,
    EnhancedShuffleNetv2_4x,
)
from nano.models.necks.ghost_pan import GhostPAN
from nano.models.heads.nanodet_head import NanoHead



# Params size (MB): 3.03
class GhostNano_3x4_m96(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        model_width = (24, 96, 192, 288, 96, 96, 96, 96, 96)
        strides = (8, 16, 32, 64)
        self.backbone = EnhancedShuffleNetv2_3x(model_width[:4])
        self.neck = GhostPAN(model_width[1:4], model_width[4:-1])
        self.head = NanoHead(model_width[4:-1], model_width[-1], strides, num_classes)
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
        model_width = (24, 48, 96, 192, 288, 96, 96, 96, 96)
        strides = (8, 16, 32)
        self.backbone = EnhancedShuffleNetv2_4x(model_width[:5])
        self.neck = GhostPAN(model_width[1:5], model_width[5:-1])
        self.head = NanoHead(model_width[5:-1], model_width[-1], strides, num_classes)
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
        model_width = (24, 96, 192, 288, 96, 96, 96, 96)
        strides = (8, 16, 32)
        self.backbone = EnhancedShuffleNetv2_3x(model_width[:4])
        self.neck = GhostPAN(model_width[1:4], model_width[4:-1])
        self.head = NanoHead(model_width[4:-1], model_width[-1], strides, num_classes)
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
        model_width = (24, 64, 128, 192, 64, 64, 64, 64)
        strides = (8, 16, 32)
        self.backbone = EnhancedShuffleNetv2_3x(model_width[:4])
        self.neck = GhostPAN(model_width[1:4], model_width[4:-1])
        self.head = NanoHead(model_width[4:-1], model_width[-1], strides, num_classes)
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
        model_width = (24, 64, 128, 192, 32, 32, 32, 32)
        strides = (8, 16, 32)
        self.backbone = EnhancedShuffleNetv2_3x(model_width[:4])
        self.neck = GhostPAN(model_width[1:4], model_width[4:-1])
        self.head = NanoHead(model_width[4:-1], model_width[-1], strides, num_classes)
        self.num_classes = num_classes
        self.strides = strides

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
