from numpy import ceil
from ._layer import CaffeLayer, parse_attribute
from caffe import params as P
import onnx


class Pool2d:
    def __init__(self, node: onnx.NodeProto, mode) -> None:
        # basic attributes
        self.node_name = node.name
        self.input_names = [str(node.input[0])]
        self.output_names = [str(node.output[0])]
        # pooling attributes
        attrs = parse_attribute(node)
        self.kernel_size = attrs["kernel_shape"]
        self.stride = attrs["strides"]
        self.padding = attrs.get("pads", [0, 0, 0, 0])
        self.ceil_mode = attrs.get("ceil_mode")
        if self.ceil_mode == 0:
            self.padding = [x - 1 for x in self.padding]
        else:
            raise NotImplementedError("XNNC do not support ceil_mode=1")
        self.mode = mode
        # assertion
        assert self.padding[0] == self.padding[2] and self.padding[1] == self.padding[3]

    def reshape(self, bottom_shapes):
        # -> top_shapes
        N, C, H, W = bottom_shapes[0]
        # ceil_mode = True by default
        H = int(ceil((H - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0])) + 1
        W = int(ceil((W - self.kernel_size[1] + 2 * self.padding[1]) / self.stride[1])) + 1
        return [
            (N, C, H, W),
        ]

    def to_proto(self):
        return CaffeLayer(
            "Pooling",
            self.node_name,
            self.input_names,
            self.output_names,
            pooling_param=dict(
                pool=self.mode,
                kernel_h=self.kernel_size[0],
                kernel_w=self.kernel_size[1],
                stride_h=self.stride[0],
                stride_w=self.stride[1],
                pad_h=self.padding[0],
                pad_w=self.padding[1],
            ),
        )._to_proto()


class Maxpool2d(Pool2d):
    def __init__(self, node: onnx.NodeProto) -> None:
        super().__init__(node, mode=P.Pooling.MAX)


class Avgpool2d(Pool2d):
    def __init__(self, node: onnx.NodeProto) -> None:
        super().__init__(node, mode=P.Pooling.AVE)


class GlobalAvgpool2d:
    def __init__(self, node: onnx.NodeProto) -> None:
        # basic attributes
        self.node_name = node.name
        self.input_names = [str(node.input[0])]
        self.output_names = [str(node.output[0])]

    def reshape(self, bottom_shapes):
        # -> top_shapes
        N, C, _, _ = bottom_shapes[0]
        return [
            (N, C, 1, 1),
        ]

    def to_proto(self):
        return CaffeLayer(
            "Pooling",
            self.node_name,
            self.input_names,
            self.output_names,
            pooling_param=dict(
                pool=P.Pooling.AVE,
                global_pooling=True,
            ),
        )._to_proto()
