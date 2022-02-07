from _layer import CaffeLayer, parse_attribute
import onnx
import torch


def constant_concat_killer(graph, constant_dict, tensor_dict):
    to_remove = []
    for node in graph.node:
        if node.op_type == "Concat":
            input_names = [str(x) for x in node.input]
            output_name = str(node.output[0])
            if all([(x in constant_dict) for x in input_names]):
                # update concat result
                input_values = [torch.tensor(constant_dict[x]) for x in input_names]
                dim = parse_attribute(node).get("axis", 1)
                concat_result = torch.cat(input_values, dim=dim)
                tensor_dict[output_name] = concat_result.numpy()
                # remove all existence
                for input_name in input_names:
                    del constant_dict[input_name]
                to_remove.append(node)
    for layer in to_remove:
        graph.node.remove(layer)


class Concat:
    def __init__(self, node: onnx.NodeProto) -> None:
        # basic attributes
        self.node_name = node.name
        self.input_names = [str(x) for x in node.input]
        self.output_names = [str(node.output[0])]
        # concat attributes
        attrs = parse_attribute(node)
        self.axis = attrs.get("axis", 1)

    def reshape(self, bottom_shapes):  # -> top_shapes
        top_shape = [x for x in bottom_shapes[0]]
        top_shape[self.axis] = sum([x[self.axis] for x in bottom_shapes])
        return [
            top_shape,
        ]

    def to_proto(self):
        return CaffeLayer(
            "Concat",
            self.node_name,
            self.input_names,
            self.output_names,
            axis=self.axis,
        )._to_proto()
