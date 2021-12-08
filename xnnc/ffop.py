from ._layer import CaffeLayer, parse_attribute
import onnx
from caffe import params as P


class Add:
    def __init__(self, node: onnx.NodeProto, shape_dict) -> None:
        # basic attributes
        self.node_name = node.name
        self.input_names = [str(x) for x in node.input]
        self.output_names = [str(node.output[0])]
        # add attributes
        self.broadcast = False
        if shape_dict[self.input_names[0]] != shape_dict[self.input_names[1]]:
            # Broadcast Add
            # For example, if bottom[0] is 4D with shape 100x3x40x60, the output
            # top[0] will have the same shape, and bottom[1] may have any of the
            # following shapes (for the given value of axis):
            #    (axis == 0 == -4) 100; 100x3; 100x3x40; 100x3x40x60
            #    (axis == 1 == -3)          3;     3x40;     3x40x60
            #    (axis == 2 == -2)                   40;       40x60
            #    (axis == 3 == -1)                                60
            # Furthermore, bottom[1] may have the empty shape (regardless of the value of
            # "axis") -- a scalar bias.
            attrs = parse_attribute(node)
            self.axis = attrs.get("axis", 0)
            self.broadcast = True
            broadcast_target = (
                0
                if shape_dict[self.input_names[0]] > shape_dict[self.input_names[1]]
                else 1
            )
            self.broadcast_bias = self.input_names[1 - broadcast_target]
            self.broadcast_target = self.input_names[broadcast_target]
            self.flatten_bias = self.broadcast_bias + "_broadcast"

    def reshape(self, bottom_shapes):  # -> top_shapes
        top_shape = [max([x[i] for x in bottom_shapes]) for i in range(4)]
        return [
            top_shape,
        ]

    def to_proto(self):
        if self.broadcast:
            layers = []
            layers.append(
                CaffeLayer(
                    "Flatten",
                    self.flatten_bias,
                    [self.broadcast_bias],
                    [self.flatten_bias],
                )._to_proto()
            )
            layers.append(
                CaffeLayer(
                    "Bias",
                    self.node_name,
                    [self.broadcast_target, self.flatten_bias],
                    self.output_names,
                    axis=self.axis,
                )._to_proto()
            )
            return layers
        else:
            return CaffeLayer(
                "Eltwise",
                self.node_name,
                self.input_names,
                self.output_names,
                operation=P.Eltwise.SUM,
            )._to_proto()
