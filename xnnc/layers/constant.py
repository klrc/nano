from ._layer import CaffeLayer
import onnx


class CaffeInput:
    def __init__(self, node: onnx.NodeProto) -> None:
        # basic attributes
        self.node_name = node.name
        self.output_names = [self.node_name]
        self.output_shape = [d.dim_value for d in node.type.tensor_type.shape.dim]

    def reshape(self, bottom_shapes=None):  # -> top_shapes
        top_shape = self.output_shape
        return [
            top_shape,
        ]

    def to_proto(self):
        return CaffeLayer(
            "Input",
            self.node_name,
            [],
            self.output_names,
            input_param=dict(
                shape=dict(dim=self.output_shape),
            )
        )._to_proto()