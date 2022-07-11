from typing import Iterable
import torch
import torch.nn as nn

from ...io.canvas import Canvas


class Probe(nn.Module):
    def __init__(self, model, forward_type="yolov5") -> None:
        super().__init__()
        self.data = []
        self.m = model
        self.forward_type = forward_type
        if model.training:
            self.train()
        else:
            self.eval()

    @staticmethod
    def trace_conv(layer: nn.Conv2d, target_channels: Iterable[int]):
        """
        :param layer:
        :param target_channels:
        :return:
        """
        if type(target_channels) is int:
            target_channels = [target_channels]
        w = layer.weight.data
        activation = w.mean(dim=-1).mean(dim=-1)[target_channels, :].mean(dim=0)
        return activation

    def forward(self, x):
        if self.forward_type == "yolov5":
            self.data = []
            input_im = x
            backbone_p3 = self.m.p0(x)
            backbone_p4 = self.m.p1(backbone_p3)
            backbone_p5 = self.m.p2(backbone_p4)
            x = self.m.u0(backbone_p5)
            x = torch.cat((x, backbone_p4), dim=1)  # cat backbone P4  # 1024x
            neck_0 = self.m.neck_0(x)
            x = self.m.u1(neck_0)
            x = torch.cat((x, backbone_p3), dim=1)  # cat backbone P3  # 512x
            neck_1 = self.m.neck_1(x)
            x = self.m.neck_1_conv(neck_1)
            x = torch.cat((x, neck_0), dim=1)  # cat head P4  # 512x
            neck_2 = self.m.neck_2(x)
            x = self.m.neck_2_conv(neck_2)
            x = torch.cat((x, backbone_p5), dim=1)  # cat head P5  # 1024x
            neck_3 = self.m.neck_3(x)
            self.set_checkpoint(neck_1)
            self.set_checkpoint(neck_2)
            self.set_checkpoint(neck_3)
            ret = self.m.detect([neck_1, neck_2, neck_3])  # Detect(P3, P4, P5)
            self.visualize(input_im)
            grid_sample = ret
            if not self.training:
                grid_sample = grid_sample[0]
            grid_ax = []
            for i, (_, _, h, w) in enumerate([neck_1.shape, neck_2.shape, neck_3.shape]):
                for _ in range(self.m.detect.na):
                    for pty in range(h):
                        for ptx in range(w):
                            grid_ax.append((i, ptx / w, pty / h))
            self.visualize_grid(3, grid_ax, grid_sample, input_im)
            return ret
        else:
            raise NotImplementedError

    def set_checkpoint(self, x: torch.Tensor):
        self.data.append(x.clone())

    def visualize(self, background):
        for i, x in enumerate(self.data):
            for j, (batch_im, batch_x) in enumerate(zip(background, x)):
                batch_x = batch_x.mean(dim=0)
                canvas = Canvas(batch_im)
                canvas.draw_heatmap(batch_x)
                canvas.show(title=f"probe_{i}_batch_{j}")

    def visualize_grid(self, num_probes, grid_ax, data, background, conf_thres=0.25):
        for j, (batch_im, batch_d) in enumerate(zip(background, data)):
            canvas_set = []
            for _ in range(num_probes):
                canvas_set.append(Canvas(batch_im))
            _, h, w = batch_im.shape
            obj_conf, cls_conf = batch_d[:, 4], batch_d[:, 5:].max(dim=-1, keepdim=True)[0]
            for (ia, ptx, pty), oc, cc in zip(grid_ax, obj_conf, cls_conf):
                if oc * cc > conf_thres:
                    canvas_set[ia].draw_point((int(ptx * w), int(pty * h)), radius=4, alpha=oc)
            for i in range(num_probes):
                canvas_set[i].show(title=f"grid_probe_{i}_batch_{j}")
