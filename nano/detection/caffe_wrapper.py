"""
    wrapper for testing propose, 
    allowing caffe model pass pytorch evaluator directly.
"""

import caffe
import torch
import torch.nn as nn


class CaffeWrapper(nn.Module):
    def __init__(self, caffemodel_path, prototxt_path, output_names, class_names, anchors) -> None:
        super().__init__()
        self.names = class_names
        self.output_names = output_names

        self.nc = len(class_names)  # number of classes
        self.no = len(class_names) + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.stride = torch.tensor((8, 16, 32))  # strides computed during build
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)  # shape(nl,na,2)
        self.register_buffer("anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)

        self.model = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def half(self):
        print("Warning: caffe model does not support weight casting, ignored.")
        return

    def float(self):
        print("Warning: caffe model does not support weight casting, ignored.")
        return

    def to(self, device):
        # cpu only for caffe.
        return

    def cuda(self):
        # cpu only for caffe.
        return

    def forward(self, img):
        # inference with yolov5-formatted output
        # input: processed image as shape (n, c, h, w)
        # returns: out, training_out
        self.model.blobs["input"].data[...] = img  # images, data
        out = self.model.forward()
        x = [out[output_name] for output_name in self.output_names]
        z = []
        for i in range(self.nl):
            # reshape
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                # make grid
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
                    self.grid[i] = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float().to(x[i].device)
                # normalize and get xywh
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))
        if self.training:
            return x
        else:
            return torch.cat(z, 1), x




if __name__ == "__main__":
    root = "release/yolox_cspm_depthwise_test"
    model = CaffeWrapper(
        caffemodel_path=f"{root}/yolox_cspm.caffemodel",
        prototxt_path=f"{root}/yolox_cspm.prototxt",
        output_names=["output.1", "output.2", "output.3"],
        class_names=["person", "two-wheeler", "car"],
        anchors=[[10.875, 14.921875], [31.1875, 53.28125], [143.0, 157.5]],
    )
    x = torch.rand(1, 3, 224, 416)
    out, train_out = model(x)
    print(out.shape)
    for p in train_out:
        print(p.shape)
