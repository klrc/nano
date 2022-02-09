import torch.nn as nn
from nano.models.backbones.enhanced_shufflenet_v2 import (
    EnhancedShuffleNetv2_3x,
    EnhancedShuffleNetv2_4x,
)
from nano.models.necks.ghost_pan import GhostPAN
from nano.models.heads.nanodet_head import NanoHead


class GhostNano_3x4_l128(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone_width, neck_width, strides = ((24, 128, 256, 512), 128, (8, 16, 32, 64))
        self.backbone = EnhancedShuffleNetv2_3x(backbone_width, s2_depth=8)
        self.neck = GhostPAN(backbone_width[1:], neck_width, len(strides))
        self.head = NanoHead(neck_width, neck_width, strides, num_classes)
        self.num_classes = num_classes
        self.strides = strides

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


class GhostNano_3x3_l128(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone_width, neck_width, strides = ((24, 128, 256, 512), 128, (8, 16, 32))
        self.backbone = EnhancedShuffleNetv2_3x(backbone_width, s2_depth=8)
        self.neck = GhostPAN(backbone_width[1:], neck_width, len(strides))
        self.head = NanoHead(neck_width, neck_width, strides, num_classes)
        self.num_classes = num_classes
        self.strides = strides

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


class GhostNano_3x4_m96(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone_width, neck_width, strides = ((24, 96, 192, 384), 96, (8, 16, 32, 64))
        self.backbone = EnhancedShuffleNetv2_3x(backbone_width, s2_depth=7)
        self.neck = GhostPAN(backbone_width[1:], neck_width, len(strides))
        self.head = NanoHead(neck_width, neck_width, strides, num_classes)
        self.num_classes = num_classes
        self.strides = strides

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


class GhostNano_4x3_m96(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone_width, neck_width, strides = ((24, 48, 96, 192, 384), 96, (8, 16, 32))
        self.backbone = EnhancedShuffleNetv2_4x(backbone_width, s2_depth=7)
        self.neck = GhostPAN(backbone_width[1:], neck_width, len(strides))
        self.head = NanoHead(neck_width, neck_width, strides, num_classes)
        self.num_classes = num_classes
        self.strides = strides

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


class GhostNano_3x3_m96(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone_width, neck_width, strides = ((24, 96, 192, 384), 96, (8, 16, 32))
        self.backbone = EnhancedShuffleNetv2_3x(backbone_width, s2_depth=7)
        self.neck = GhostPAN(backbone_width[1:], neck_width, len(strides))
        self.head = NanoHead(neck_width, neck_width, strides, num_classes)
        self.num_classes = num_classes
        self.strides = strides

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


class GhostNano_3x4_s64(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone_width, neck_width, strides = ((24, 64, 128, 256), 64, (8, 16, 32, 64))
        self.backbone = EnhancedShuffleNetv2_3x(backbone_width, s2_depth=6)
        self.neck = GhostPAN(backbone_width[1:], neck_width, len(strides))
        self.head = NanoHead(neck_width, neck_width, strides, num_classes)
        self.num_classes = num_classes
        self.strides = strides

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


class GhostNano_3x3_s64(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone_width, neck_width, strides = ((24, 64, 128, 256), 64, (8, 16, 32))
        self.backbone = EnhancedShuffleNetv2_3x(backbone_width, s2_depth=6)
        self.neck = GhostPAN(backbone_width[1:], neck_width, len(strides))
        self.head = NanoHead(neck_width, neck_width, strides, num_classes)
        self.num_classes = num_classes
        self.strides = strides

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x
