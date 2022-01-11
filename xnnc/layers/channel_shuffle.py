from ._layer import CaffeLayer


class ShuffleChannel:
    def __init__(self, node, constant_dict) -> None:
        # basic attributes
        super().__init__()
        self.node_name = node.name
        self.input_names = [str(node.input[0])]
        self.output_names = [str(node.output[0])]
        # shuffle_channel attributes
        self.groups = constant_dict[node.input[1]]

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
        # XNNC mResize layer definition.
        txt_proto = ""
        txt_proto += "layer {\n"
        txt_proto += '  name: "{}"\n'.format(self.node_name)
        txt_proto += '  type: "CppCustom"\n'
        txt_proto += '  bottom: "{}"\n'.format(self.input_names[0])
        txt_proto += '  top: "{}"\n'.format(self.output_names[0])
        txt_proto += "  cpp_custom_param {\n"
        txt_proto += '    module: "XnncShuffleChannel"\n'
        txt_proto += '    param_map_str: "group:{}"\n'.format(self.groups)
        txt_proto += "  }\n"
        txt_proto += "}\n"
        return txt_proto
