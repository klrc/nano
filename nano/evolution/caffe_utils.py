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


def get_graph(onnx_path):
    model = onnx.load(onnx_path)
    model = shape_inference.infer_shapes(model)
    graph = model.graph
    return graph


def convert_to_caffe(graph, prototxt_path, caffemodel_path):
    input_tensors = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    caffe_layers = []
    for node in graph.node:
        node: onnx.NodeProto
        if node.op_type == "Conv":
            # register attributes
            node_name = node.name
            print(node)
            weight_name = node.inputs[1]
            input_names = [str(node.inputs[0])]
            output_names = [str(node.outputs[0])]
            kernel_size = node.attrs["kernel_shape"]
            stride = node.attrs["strides"]
            padding = node.attrs.get("pads", [0, 0, 0, 0])
            groups = node.attrs.get("group", 1)
            dilation = node.attrs.get("dilations", [1, 1])
            bias = len(node.inputs) > 2
            weight_shape = node.input_tensors[weight_name].shape
            out_channels = weight_shape[0]
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
                )
            )
        elif node.op_type == "Relu":
            node_name = node.name
            input_names = [str(node.inputs[0])]
            output_names = [str(node.outputs[0])]
            inplace = input_names[0] == output_names[0]
            caffe_layers.append(
                operators.relu(
                    node_name,
                    input_names,
                    output_names,
                    inplace,
                )
            )
        elif node.op_type == "Concat":
            node_name = node.name
            input_names = [str(i) for i in node.inputs]
            output_names = str(node.outputs[0])
            axis = node.attrs.get("axis", 1)
            caffe_layers.append(
                operators.concat(
                    node_name,
                    input_names,
                    output_names,
                    axis,
                )
            )
        elif node.op_type == "MaxPool":
            node_name = node.name
            input_names = str(node.inputs[0])
            output_names = str(node.outputs[0])
            kernel_size = node.attrs["kernel_shape"]
            stride = node.attrs.get("strides", [1, 1])
            padding = node.attrs.get("pads", [0, 0, 0, 0])
            caffe_layers.append(
                operators.maxpool2d(
                    node_name,
                    input_names,
                    output_names,
                    kernel_size,
                    stride,
                    padding,
                )
            )
        elif node.op_type == "AveragePool":
            node_name = node.name
            input_names = str(node.inputs[0])
            output_names = str(node.outputs[0])
            kernel_size = node.attrs["kernel_shape"]
            stride = node.attrs.get("strides", [1, 1])
            padding = node.attrs.get("pads", [0, 0, 0, 0])
            caffe_layers.append(
                operators.avgpool2d(
                    node_name,
                    input_names,
                    output_names,
                    kernel_size,
                    stride,
                    padding,
                )
            )
    net = caffe_pb2.NetParameter()
    for id, layer in enumerate(caffe_layers):
        caffe_layers[id] = layer._to_proto()
    net.layer.extend(caffe_layers)

    with open(prototxt_path, 'w') as f:
        print(net, file=f)

    # caffe.set_mode_cpu()
    # deploy = prototxt_save_path
    # net = caffe.Net(deploy,
    #                 caffe.TEST)

    # for id, node in enumerate(graph.nodes):
    #     op_type = node.op_type
    #     inputs = node.inputs
    #     inputs_tensor = node.input_tensors
    #     input_non_exist_flag = False
    #     if op_type not in wlr._ONNX_NODE_REGISTRY:
    #         err.unsupported_op(node)
    #         continue
    #     converter_fn = wlr._ONNX_NODE_REGISTRY[op_type]
    #     converter_fn(net, node, graph, err)

    # net.save(caffe_model_save_path)
    # return net
    # return


def compare_onnx_and_caffe(onnx_path, prototxt_path, caffemodel_path):
    return


def onnx_to_caffe(onnx_path):
    prototxt_path = onnx_path.replace(".onnx", ".prototxt")
    caffemodel_path = onnx_path.replace(".onnx", ".caffemodel")
    graph = get_graph(onnx_path)
    convert_to_caffe(graph, prototxt_path, caffemodel_path)
    compare_onnx_and_caffe(onnx_path, prototxt_path, caffemodel_path)
    return prototxt_path, caffemodel_path
