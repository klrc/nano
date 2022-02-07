import caffe
from numpy import ceil
from _layer import CaffeLayer, parse_attribute
import onnx
import numpy as np


class Conv2d:
    def __init__(self, node: onnx.NodeProto, tensor_dict, shape_dict) -> None:
        # basic attributes
        self.node_name = node.name
        self.input_names = [str(node.input[0])]
        self.output_names = [str(node.output[0])]
        # conv2d attributes
        attrs = parse_attribute(node)
        self.kernel_size = attrs["kernel_shape"]
        self.stride = attrs["strides"]
        self.padding = attrs.get("pads", [0, 0, 0, 0])
        self.groups = attrs.get("group", 1)
        self.dilation = attrs.get("dilations", [1, 1])
        self.bias = len(node.input) > 2
        # register params
        self.weight_params = tensor_dict[str(node.input[1])]
        if self.bias:
            self.bias_params = tensor_dict[str(node.input[2])]
        # parse channels
        weight_shape = self.weight_params.shape
        self.out_channels = weight_shape[0]
        self.in_channels = shape_dict[self.input_names[0]][1]
        # assertion
        assert self.padding[0] == self.padding[2] and self.padding[1] == self.padding[3]
        if self.groups > 1:
            # XNNC do not support group convolution layer except depth-wise convolution
            # so we check the groups here
            assert (
                self.groups == self.out_channels == self.in_channels
            ), f"group conv not supported as in_channels={self.in_channels}, groups={self.groups}, out_channels={self.out_channels}"

    def reshape(self, bottom_shapes):
        # -> top_shapes
        N, _, H, W = bottom_shapes[0]
        C = self.out_channels
        # ceil_mode = True by default
        H = int(ceil((H - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0])) + 1
        W = int(ceil((W - self.kernel_size[1] + 2 * self.padding[1]) / self.stride[1])) + 1
        return [
            (N, C, H, W),
        ]

    def to_proto(self):
        return CaffeLayer(
            "Convolution",
            self.node_name,
            self.input_names,
            self.output_names,
            kernel_h=self.kernel_size[0],
            kernel_w=self.kernel_size[1],
            stride_h=self.stride[0],
            stride_w=self.stride[1],
            group=self.groups,
            pad_h=self.padding[0],
            pad_w=self.padding[1],
            num_output=self.out_channels,
            dilation=self.dilation[0],
            bias_term=self.bias,
        )._to_proto()

    def inject_params(self, net: caffe.Net):
        np.copyto(
            net.params[self.node_name][0].data,
            self.weight_params,
            casting="same_kind",
        )
        if hasattr(self, "bias_params"):
            np.copyto(
                net.params[self.node_name][1].data,
                self.bias_params,
                casting="same_kind",
            )
