from ._layer import XNNCLayer, parse_attribute
import onnx


class Resize(XNNCLayer):
    def __init__(self, node, tensor_dict) -> None:
        super().__init__()
        # basic attributes
        self.node_name = node.name
        self.input_names = [str(node.input[0])]
        self.output_names = [str(node.output[0])]
        # resize attributes
        self.scale_factor = tensor_dict[str(node.input[2])][2:]
        # XNNC layer settings
        self.param_str =(
            '{' + "\'ScaleX\':{}, \'ScaleY\':{}".format(self.scale_factor[0], self.scale_factor[1]) + '}'
        )
        self.xnnc_rep = 'XNNCTypeResize'

    def reshape(self, bottom_shapes):  # -> top_shapes
        N, C, H, W = bottom_shapes[0]
        H = H * self.scale_factor[0]
        W = W * self.scale_factor[1]
        return [
            (N, C, H, W),
        ]

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
        txt_proto += (
            '    param_map_str: "scaleX:{} scaleY:{} align_corners:1"\n'.format(
                self.scale_factor[0], self.scale_factor[1]
            )
        )
        txt_proto += "  }\n"
        txt_proto += "}\n"
        return txt_proto
