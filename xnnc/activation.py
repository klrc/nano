from ._layer import CaffeLayer, parse_attribute
import onnx


class Relu:
    def __init__(self, node: onnx.NodeProto) -> None:
        # basic attributes
        self.node_name = node.name
        self.input_names = [str(node.input[0])]
        self.output_names = [str(node.output[0])]
        # relu attributes
        self.inplace = self.input_names[0] == self.output_names[0]

    def reshape(self, bottom_shapes):  # -> top_shapes
        return bottom_shapes

    def to_proto(self):
        return CaffeLayer(
            "ReLU",
            self.node_name,
            self.input_names,
            self.output_names,
            inplace=self.inplace,
        )._to_proto()


class LeakyRelu(Relu):
    def __init__(self, node: onnx.NodeProto) -> None:
        super().__init__(node)
        # LeakyRelu attrbutes
        attrs = parse_attribute(node)
        self.alpha = attrs["alpha"]

    def to_proto(self):
        return CaffeLayer(
            "ReLU",
            self.node_name,
            self.input_names,
            self.output_names,
            inplace=self.inplace,
            relu_param=dict(
                negative_slope=self.alpha,
            ),
        )._to_proto()


class Relu6:
    def __init__(self, node, constant_dict) -> None:
        super().__init__()
        # basic attributes
        self.node_name = node.name
        self.input_names = [str(node.input[0])]
        self.output_names = [str(node.output[0])]
        # relu6 attributes
        self.inplace = self.input_names[0] == self.output_names[0]
        clip_min = float(constant_dict[str(node.input[1])])
        clip_max = float(constant_dict[str(node.input[2])])
        assert clip_min == 0 and clip_max == 6, NotImplementedError()

    def reshape(self, bottom_shapes):  # -> top_shapes
        return bottom_shapes

    def shadow_proto(self):
        return CaffeLayer(
            "ReLU",
            self.node_name,
            self.input_names,
            self.output_names,
        )._to_proto()

    def to_proto(self):
        txt_proto = ""
        txt_proto += "layer {\n"
        txt_proto += '  name: "{}"\n'.format(self.node_name)
        txt_proto += '  type: "ReLU"\n'
        txt_proto += '  bottom: "{}"\n'.format(self.input_names[0])
        txt_proto += '  top: "{}"\n'.format(self.output_names[0])
        txt_proto += "  relu_param {\n"
        txt_proto += "    relu6_enable: 1\n"
        txt_proto += "  }\n"
        txt_proto += "}\n"
        return txt_proto
