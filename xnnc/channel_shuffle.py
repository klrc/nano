from ._layer import XNNCLayer, parse_attribute
import onnx

# layer {
#   name: "shuffle2"
#   type: "CppCustom"
#   bottom: "resx2_conv1"
#   top: "shuffle2"
#   cpp_custom_param {
#     module: "XnncShuffleChannel"
#     param_map_str: "group:3 "
#   }
# }
class ShuffleChannel(XNNCLayer):
    def __init__(self, node, constant_dict) -> None:
        # basic attributes
        super().__init__()
        self.node_name = node.name
        self.input_names = [str(node.input[0])]
        self.output_names = [str(node.output[0])]
        # slice attributes
        self.groups = constant_dict[node.input[1]]

    def reshape(self, bottom_shapes):  # -> top_shapes
        return bottom_shapes

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
