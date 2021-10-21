import caffe
import onnx
from onnx import (
    shape_inference,
    numpy_helper,
    ValueInfoProto,
    AttributeProto,
    GraphProto,
    NodeProto,
    TensorProto,
    TensorShapeProto,
)
from caffe.proto import caffe_pb2
from . import operators
from .consistency_check import compare_onnx_and_caffe
from onnx import numpy_helper, ValueInfoProto, AttributeProto, GraphProto, NodeProto, TensorProto, TensorShapeProto
from typing import Any, Text, Iterable, List, Dict, Sequence, Optional, Tuple, Union
from typing_extensions import Protocol
import numpy as np


def _convertAttributeProto(onnx_arg):
    """
    Convert an ONNX AttributeProto into an appropriate Python object
    for the type.
    NB: Tensor attribute gets returned as numpy array
    """
    if onnx_arg.HasField("f"):
        return onnx_arg.f
    elif onnx_arg.HasField("i"):
        return onnx_arg.i
    elif onnx_arg.HasField("s"):
        return onnx_arg.s
    elif onnx_arg.HasField("t"):
        return numpy_helper.to_array(onnx_arg.t)
    elif len(onnx_arg.floats):
        return list(onnx_arg.floats)
    elif len(onnx_arg.ints):
        return list(onnx_arg.ints)
    elif len(onnx_arg.strings):
        return list(onnx_arg.strings)
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(onnx_arg))


class Attributes(Dict[Text, Any]):
    @staticmethod
    def from_onnx(args):  # type: (Iterable[AttributeProto]) -> Attributes
        d = Attributes()
        for arg in args:
            d[arg.name] = _convertAttributeProto(arg)
        return d


def get_graph(onnx_path):
    model = onnx.load(onnx_path)
    model = shape_inference.infer_shapes(model)
    graph = model.graph
    return graph


def convert_to_caffe(graph, prototxt_path, caffemodel_path):
    input_tensors = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    net = caffe_pb2.NetParameter()
    caffe_layers = net.layer
    buffered_params = {}
    blob_channels = {}
    # register inputs
    for node in graph.input:
        node_name = node.name
        if node_name in input_tensors:
            continue
        output_names = [node_name]
        output_shape = [d.dim_value for d in node.type.tensor_type.shape.dim]
        blob_channels[node_name] = output_shape[1]
        caffe_layers.append(
            operators.input_node(
                node_name,
                output_names,
                output_shape,
            )._to_proto()
        )
    # register nodes
    for node in graph.node:
        node: onnx.NodeProto
        attrs = Attributes.from_onnx(node.attribute)
        if node.op_type == "Conv":
            # register attributes
            node_name = node.name
            weight_name = node.input[1]
            input_names = [str(node.input[0])]
            output_names = [str(node.output[0])]
            kernel_size = attrs["kernel_shape"]
            stride = attrs["strides"]
            padding = attrs.get("pads", [0, 0, 0, 0])
            groups = attrs.get("group", 1)
            dilation = attrs.get("dilations", [1, 1])
            bias = len(node.input) > 2
            weight_shape = input_tensors[weight_name].shape
            out_channels = weight_shape[0]
            blob_channels[output_names[0]] = out_channels
            buffered_params[(node_name, 0)] = input_tensors[weight_name]
            if bias:
                buffered_params[(node_name, 1)] = input_tensors[str(node.input[2])]
            # append layer
            caffe_layers.append(
                operators.conv2d(
                    node_name,
                    input_names,
                    output_names,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    groups,
                    dilation,
                    bias,
                )._to_proto()
            )
        elif node.op_type == "Relu":
            node_name = node.name
            input_names = [str(node.input[0])]
            output_names = [str(node.output[0])]
            inplace = input_names[0] == output_names[0]
            blob_channels[output_names[0]] = blob_channels[input_names[0]]
            caffe_layers.append(
                operators.relu(
                    node_name,
                    input_names,
                    output_names,
                    inplace,
                )._to_proto()
            )
        elif node.op_type == "Concat":
            node_name = node.name
            input_names = [str(x) for x in node.input]
            output_names = [str(node.output[0])]
            blob_channels[output_names[0]] = sum([blob_channels[x] for x in input_names])
            axis = attrs.get("axis", 1)
            caffe_layers.append(
                operators.concat(
                    node_name,
                    input_names,
                    output_names,
                    axis,
                )._to_proto()
            )
        elif node.op_type == "MaxPool":
            node_name = node.name
            input_names = str(node.input[0])
            output_names = str(node.output[0])
            kernel_size = attrs["kernel_shape"]
            stride = attrs.get("strides", [1, 1])
            padding = attrs.get("pads", [0, 0, 0, 0])
            blob_channels[output_names[0]] = blob_channels[input_names[0]]
            caffe_layers.append(
                operators.maxpool2d(
                    node_name,
                    input_names,
                    output_names,
                    kernel_size,
                    stride,
                    padding,
                )._to_proto()
            )
        elif node.op_type == "AveragePool":
            node_name = node.name
            input_names = str(node.input[0])
            output_names = str(node.output[0])
            kernel_size = attrs["kernel_shape"]
            stride = attrs.get("strides", [1, 1])
            padding = attrs.get("pads", [0, 0, 0, 0])
            blob_channels[output_names[0]] = blob_channels[input_names[0]]
            caffe_layers.append(
                operators.avgpool2d(
                    node_name,
                    input_names,
                    output_names,
                    kernel_size,
                    stride,
                    padding,
                )._to_proto()
            )
        elif node.op_type == "Add":
            node_name = node.name
            input_names = [str(x) for x in node.input]
            output_names = [str(node.output[0])]
            if blob_channels[input_names[0]] != blob_channels[input_names[1]]:
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
                broadcast_target = 0 if blob_channels[input_names[0]] > blob_channels[input_names[1]] else 1
                broadcast_bias = input_names[1 - broadcast_target]
                broadcast_target = input_names[broadcast_target]
                flatten_bias = broadcast_bias + "_flatten"
                axis = attrs.get("axis", 0)
                blob_channels[output_names[0]] = blob_channels[broadcast_target]
                caffe_layers.append(
                    operators.flatten(
                        node_name + "_flatten",
                        [broadcast_bias],
                        [flatten_bias],
                    )._to_proto()
                )
                caffe_layers.append(
                    operators.bias(
                        node_name,
                        [broadcast_target, flatten_bias],
                        output_names,
                        axis=axis,
                    )._to_proto()
                )
            else:
                blob_channels[output_names[0]] = blob_channels[input_names[0]]
                caffe_layers.append(
                    operators.add(
                        node_name,
                        input_names,
                        output_names,
                    )._to_proto()
                )
        elif node.op_type == "Resize":
            node_name = node.name
            input_names = [str(node.input[0])]
            output_names = [str(node.output[0])]
            scale_factor = input_tensors.get(str(node.input[2]))[2:]
            in_channels = blob_channels[input_names[0]]
            blob_channels[output_names[0]] = blob_channels[input_names[0]]
            mode = attrs["mode"]
            if str(mode, encoding="gbk") == "linear" and scale_factor[0] > 1 and scale_factor[1] > 1:
                caffe_layers.append(
                    operators.upsample_bilinear2d(
                        node_name,
                        input_names,
                        output_names,
                        in_channels,
                        scale_factor,
                    )._to_proto()
                )
            elif str(mode, encoding="gbk") == "linear" and scale_factor[0] == 0.5 and scale_factor[1] == 0.5:
                kernel_size = [2, 2]
                stride = [2, 2]
                padding = [0, 0, 0, 0]
                caffe_layers.append(
                    operators.avgpool2d(
                        node_name,
                        input_names,
                        output_names,
                        kernel_size,
                        stride,
                        padding,
                    )._to_proto()
                )
            else:
                raise NotImplementedError
        else:
            # print(f'warning: ignored {node.op_type} node {node.name}')
            pass

    with open(prototxt_path, "w") as f:
        f.write(str(net))

    caffe.set_mode_cpu()
    net = caffe.Net(prototxt_path, caffe.TEST)
    for (node_name, order), W in buffered_params.items():
        np.copyto(net.params[node_name][order].data, W, casting="same_kind")
    net.save(caffemodel_path)
    return


def onnx_to_caffe(onnx_path):
    prototxt_path = onnx_path.replace(".onnx", ".prototxt")
    caffemodel_path = onnx_path.replace(".onnx", ".caffemodel")
    graph = get_graph(onnx_path)
    convert_to_caffe(graph, prototxt_path, caffemodel_path)
    compare_onnx_and_caffe(onnx_path, prototxt_path, caffemodel_path)
    return prototxt_path, caffemodel_path
