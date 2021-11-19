import torch
# from model.yolov5s import Xyolov5s
from yolox import YoloxShuffleNetES as Xyolov5s


def attempt_create(pretrained=True):
    model = Xyolov5s() #pretrained=pretrained)
    # model.as_relu()
    # model.as_bilinear() #I will test later
    return model


def attempt_load(weights, device):
    model = attempt_create(pretrained=False)
    if weights.endswith('.pt') or weights.endswith('.pth'):
        print('load:', weights)
        ckpt = torch.load(weights)
        # print(ckpt)
        # input()
        # model.load_state_dict(ckpt['ema' if ckpt.get('ema') else 'model'].state_dict())
        model.load_state_dict(ckpt)
    return model.float().eval().to(device) #.fuse()
