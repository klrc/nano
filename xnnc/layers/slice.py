from ._layer import CaffeLayer, parse_attribute
import onnx
import math
from onnx import helper


def slice_killer(graph):
    # merge slice layers with same input data
    slice_groups = {}
    for node in graph.node:
        if node.op_type == "Slice":
            data = str(node.input[0])
            if data not in slice_groups:
                slice_groups[data] = []
            slice_groups[data].append(node)
    for data, layers in slice_groups.items():
        # generate a caffe slice layer
        # data starts ends axes steps
        data = str(layers[0].input[0]) # data
        axes = layers[0].input[3]  # axes
        assert all([data == layer.input[0] for layer in layers])
        inputs = [data]  # input blobs
        outputs = [str(layer.output[0]) for layer in layers]  # output blobs
        slice_points = [layer.input[2] for layer in layers]  # ends
        slice_points = sorted(slice_points)
        slice_points = slice_points[:-1]  # remove last end
        slice_layer = helper.make_node(
            "UnifiedSlice",
            inputs=inputs,
            outputs=outputs,
            axes=axes,
            slice_points=slice_points,
        )
        # kill all slice layers
        first_occur = None
        for i, node in enumerate(graph.node):
            if node == layers[0]:
                first_occur = i
        assert first_occur is not None
        for layer in layers:
            graph.node.remove(layer)
        # insert layer at the first occurred position
        graph.node.insert(first_occur, slice_layer)


class Slice:
    def __init__(self, node, constant_dict) -> None:
        super().__init__()
        # basic attributes
        self.node_name = node.name
        self.input_names = [str(node.input[0])]
        self.output_names = [str(x) for x in node.output]
        # slice attributes
        attrs = parse_attribute(node)
        self.axis = constant_dict[str(int(attrs["axes"]))][0]
        self.slice_points = [constant_dict[str(int(x))][0] for x in attrs["slice_points"]]
        

    def reshape(self, bottom_shapes):  # -> top_shapes
        bottom_shape = list(bottom_shapes[0])
        # TODO： FIX THIS
        # top_shapes = 
        # top_shape[self.dim] = top_shape[self.dim] // self.chunks
        # top_shapes = [top_shape for _ in range(self.chunks)]
        # return top_shapes

    def shadow_proto(self):
        return CaffeLayer(
            "Slice",
            self.node_name,
            self.input_names,
            self.output_names,
            slice_param=dict(
                axis=self.axes,
                slice_point=[self.starts, self.ends],
            ),
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
        txt_proto += '    module: "mSlice"\n'
        txt_proto += '    param_map_str: "scaleX:{} scaleY:{} align_corners:1"\n'.format(int(self.scale_factor[0]), int(self.scale_factor[1]))
        txt_proto += "  }\n"
        txt_proto += "}\n"
        return txt_proto
