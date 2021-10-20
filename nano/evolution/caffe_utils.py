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
from torch.autograd.grad_mode import no_grad


def get_graph(onnx_path):
    model = onnx.load(onnx_path)
    model = shape_inference.infer_shapes(model)
    graph = model.graph
    return graph


def convert_to_caffe(graph, prototxt_path, caffemodel_path):
    input_tensors = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    for node in graph.node:
        node: onnx.NodeProto
        if node.op_type == 'Constant':
            print(node)
    return


def compare_onnx_and_caffe(onnx_path, prototxt_path, caffemodel_path):
    return


def onnx_to_caffe(onnx_path):
    prototxt_path = onnx_path.replace(".onnx", ".prototxt")
    caffemodel_path = onnx_path.replace(".onnx", ".caffemodel")
    graph = get_graph(onnx_path)
    convert_to_caffe(graph, prototxt_path, caffemodel_path)
    compare_onnx_and_caffe(onnx_path, prototxt_path, caffemodel_path)
    return prototxt_path, caffemodel_path
