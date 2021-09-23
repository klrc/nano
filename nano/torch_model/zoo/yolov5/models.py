import torch
import torch.nn as nn
from .common import Focus, Conv, C3, SPP, Detect

class CSPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage_1 = nn.Sequential(
            Focus(3, 32),
            Conv(32, 64, 3, 2),
            C3(64, 64, 1),
            Conv(64, 128, 3, 2),
            C3(128, 128, 3),
        )
        self.stage_2 = nn.Sequential(
            Conv(128, 256, 3, 2),
            C3(256, 256, 3),
        )
        self.stage_3 = nn.Sequential(
            Conv(256, 512, 3, 2),
            SPP(512),
            C3(512, 512, 1, False),
            Conv(512, 256, 1, 1),
        )
    
    def forward(self, x):
        x1 = self.stage_1(x)  # (1, 128, .., ..)
        x2 = self.stage_2(x1)  # (1, 256, .., ..)
        x3 = self.stage_3(x2)  # (1, 256, .., ..)
        return x1, x2, x3


class PAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_p1 = nn.Upsample(None, 2, 'nearest')
        self.up_p2 = nn.Upsample(None, 2, 'nearest')
        self.conv_p3 = Conv(128, 128, 3, 2)
        self.conv_p4 = Conv(256, 256, 3, 2)
        self.head_p1 = nn.Sequential(
            C3(512, 256, 1, False),
            Conv(256, 128, 1, 1),
        )
        self.c3_p2 = C3(256, 128, 1, False)
        self.c3_p3 = C3(256, 256, 1, False)
        self.c3_p4 = C3(512, 512, 1, False)

    def forward(self, x):
        x1, x2, x3 = x
        f1 = self.head_p1(torch.cat((self.up_p1(x3), x2), dim=1))  # (1, 128, .., ..)
        f2 = self.c3_p2(torch.cat((self.up_p2(f1), x1), dim=1))  # (1, 128, .., ..)
        f3 = self.c3_p3(torch.cat((self.conv_p3(f2), f1), dim=1))  # (1, 256, .., ..)
        f4 = self.c3_p4(torch.cat((self.conv_p4(f3), x3), dim=1))  # (1, 512, .., ..)
        return f2, f3, f4


class YOLO_V5(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.mixer = PAN()
        self.detect = Detect(nc=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.mixer(x)
        x = self.detect(x)
        return x


def yolov5s(num_classes=80, **kwargs):
    backbone = CSPNet()
    model = YOLO_V5(backbone, num_classes, **kwargs)
    return model