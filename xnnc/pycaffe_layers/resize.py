from _layer import CaffeLayer, parse_attribute
import onnx
import math


class Resize:
    def __init__(self, node, tensor_dict, shape_dict) -> None:
        super().__init__()
        # basic attributes
        self.node_name = node.name
        self.input_names = [str(node.input[0])]
        self.output_names = [str(node.output[0])]
        # resize attributes
        attrs = parse_attribute(node)
        self.mode = str(attrs["mode"], encoding="gbk")
        self.scale_factor = tensor_dict[str(node.input[2])][2:]
        self.in_channels = shape_dict[self.input_names[0]][1]
        # assertions
        assert self.scale_factor[0] > 1 and self.scale_factor[1] > 1

    def reshape(self, bottom_shapes):  # -> top_shapes
        N, C, H, W = bottom_shapes[0]
        H = H * self.scale_factor[0]
        W = W * self.scale_factor[1]
        return [
            (N, C, H, W),
        ]

    def shadow_proto(self):
        if self.mode == "linear":
            kernel_h = int(2 * self.scale_factor[0] - self.scale_factor[0] % 2)
            kernel_w = int(2 * self.scale_factor[1] - self.scale_factor[1] % 2)
            stride_h = int(self.scale_factor[0])
            stride_w = int(self.scale_factor[1])
            pad_h = int(math.ceil((self.scale_factor[0] - 1) / 2.0))
            pad_w = int(math.ceil((self.scale_factor[1] - 1) / 2.0))
            return CaffeLayer(
                "Deconvolution",
                "_" + self.node_name,
                self.input_names,
                self.output_names,
                convolution_param=dict(
                    num_output=self.in_channels,
                    kernel_h=kernel_h,
                    kernel_w=kernel_w,
                    stride_h=stride_h,
                    stride_w=stride_w,
                    pad_h=pad_h,
                    pad_w=pad_w,
                    group=self.in_channels,
                    bias_term=False,
                    weight_filler=dict(type="bilinear"),
                ),
            )._to_proto()
        else:
            raise NotImplementedError

    def to_proto(self):
        # XNNC mResize layer definition.
        txt_proto = ""
        txt_proto += "layer {\n"
        txt_proto += '  name: "{}"\n'.format(self.node_name)
        txt_proto += '  type: "CppCustom"\n'
        txt_proto += '  bottom: "{}"\n'.format(self.input_names[0])
        txt_proto += '  top: "{}"\n'.format(self.output_names[0])
        txt_proto += "  cpp_custom_param {\n"
        txt_proto += '    module: "mResize"\n'
        txt_proto += '    param_map_str: "scaleX:{} scaleY:{} align_corners:1"\n'.format(
            int(self.scale_factor[0]), int(self.scale_factor[1])
        )
        txt_proto += "  }\n"
        txt_proto += "}\n"
        return txt_proto
