from ._layer import XNNCLayer, parse_attribute
import onnx

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
        axes = layers[0].input[3]  # axes
        inputs = [str(layers[0].input[0])]  # input blobs
        outputs = [str(x.output[0]) for x in layers]  # output blobs
        slice_points = [x.input[2] for x in layers]  # ends
        slice_points = slice_points[:-1]  # remove last end
        slice_layer = helper.make_node(
            "_Slice",
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


class Slice(XNNCLayer):
    def __init__(self, node, constant_dict) -> None:
        super().__init__()
        # basic attributes
        self.node_name = node.name
        self.input_names = [str(node.input[0])]
        self.output_names = [str(x) for x in node.output]
        # slice attributes
        attrs = parse_attribute(node)
        axis = attrs["axes"]
        slice_points = attrs["slice_points"]
        self.dim = constant_dict[str(int(axis))][0]
        self.chunks = len(slice_points) + 1
        # assertion
        slice_points = [constant_dict[str(int(x))][0] for x in slice_points]
        # XNNC layer settings
        self.param_str = (
            '{' + "\'chunks\':{}, \'dim\':{}".format(self.chunks, self.dim) + '}'
        )
        self.xnnc_rep = "XNNCTypeSlice"

    def reshape(self, bottom_shapes):  # -> top_shapes
        top_shape = [x for x in bottom_shapes[0]]
        top_shape[self.dim] = top_shape[self.dim] // self.chunks
        top_shapes = [top_shape for _ in range(self.chunks)]
        return top_shapes

    def to_proto(self):
        # XNNC mResize layer definition.
        txt_proto = ""
        txt_proto += "layer {\n"
        txt_proto += '  name: "{}"\n'.format(self.node_name)
        txt_proto += '  type: "CppCustom"\n'
        txt_proto += '  bottom: "{}"\n'.format(self.input_names[0])
        for output_name in self.output_names:
            txt_proto += '  top: "{}"\n'.format(output_name)
        txt_proto += "  cpp_custom_param {\n"
        txt_proto += '    module: "slice"\n'
        txt_proto += '    param_map_str: "chunks:{} dim:{}"\n'.format(
            self.chunks,
            self.dim,
        )
        txt_proto += "  }\n"
        txt_proto += "}\n"
        return txt_proto
