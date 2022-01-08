import torch
import torch.nn as nn
from nano.models.backbones.enhanced_shufflenet_v2 import (
    EnhancedShuffleNetv2_x4_l,
    EnhancedShuffleNetv2_x3_m,
    EnhancedShuffleNetv2_x3_s,
)
from nano.models.necks.csp_pan import (
    CSPPAN_4x3,
    CSPPAN_3x3,
)
from nano.models.heads.yolox_head import YoloXHead


class ESyolox_4x3_l96(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = EnhancedShuffleNetv2_x4_l()
        self.neck = CSPPAN_4x3([64, 128, 256, 512], 96)
        self.head = YoloXHead((96, 96, 96), 96, num_classes=num_classes)
        self.num_classes = num_classes
        self.strides = (8, 16, 32)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x
        

class ESyolox_4x3_l64(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = EnhancedShuffleNetv2_x4_l()
        self.neck = CSPPAN_4x3([64, 128, 256, 512], 64)
        self.head = YoloXHead((64, 64, 64), 64, num_classes=num_classes)
        self.num_classes = num_classes
        self.strides = (8, 16, 32)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


class ESyolox_3x3_m96(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = EnhancedShuffleNetv2_x3_m()
        self.neck = CSPPAN_3x3([96, 192, 288], 96)
        self.head = YoloXHead((96, 96, 96), 96, num_classes=num_classes)
        self.num_classes = num_classes
        self.strides = (8, 16, 32)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


class ESyolox_3x3_m48(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = EnhancedShuffleNetv2_x3_m()
        self.neck = CSPPAN_3x3([96, 192, 288], 48)
        self.head = YoloXHead((48, 48, 48), 48, num_classes=num_classes)
        self.num_classes = num_classes
        self.strides = (8, 16, 32)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

class ESyolox_3x3_s32(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = EnhancedShuffleNetv2_x3_s()
        self.neck = CSPPAN_3x3([64, 128, 192], 32)
        self.head = YoloXHead((32, 32, 32), 32, num_classes=num_classes)
        self.num_classes = num_classes
        self.strides = (8, 16, 32)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

if __name__ == "__main__":
    from nano.models.utils import check_size

    model = ESyolox_4x3_l96(3)
    x = torch.rand(4, 3, 224, 416)
    for y in model(x):
        print(y.shape)
    check_size(model)
