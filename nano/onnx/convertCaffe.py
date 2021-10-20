# -*- coding: utf-8 -*-
from __future__ import print_function
import sys

import os,sys
# caffe_root='/opt/caffe/python'
# os.chdir(caffe_root)
# sys.path.insert(0,caffe_root)

import caffe
import onnx
import numpy as np
from caffe.proto import caffe_pb2
caffe.set_mode_cpu()
from onnx2caffe._transformers import *
from onnx2caffe._graph import Graph

import onnx2caffe._operators as cvt
import onnx2caffe._weightloader as wlr
from onnx2caffe._error_utils import ErrorHandling
from collections import OrderedDict
from onnx import shape_inference

from modelComparator import compareOnnxAndCaffe

transformers = [
    TransposeKiller(),
    ConstantsToInitializers(),
    ConvAddFuser(),
    MatmulAddFuser(),
    UnsqueezeFuser(),
]

def convertToCaffe(graph,opset_version, prototxt_save_path, caffe_model_save_path):

    exist_edges = []
    layers = []
    exist_nodes = []
    err = ErrorHandling()
    for i in graph.inputs:
        edge_name = i[0]
        input_layer = cvt.make_input(i,opset_version)
        layers.append(input_layer)
        exist_edges.append(i[0])
        graph.channel_dims[edge_name] = graph.shape_dict[edge_name][1]


    for id, node in enumerate(graph.nodes):
        print(node.name, node.op_type)
        node_name = node.name
        op_type = node.op_type
        inputs = node.inputs
        inputs_tensor = node.input_tensors
        input_non_exist_flag = False

        for inp in inputs:
            if inp not in exist_edges and inp not in inputs_tensor:
                input_non_exist_flag = True
                break
        if input_non_exist_flag:
            continue

        if op_type not in cvt._ONNX_NODE_REGISTRY:
            err.unsupported_op(node)
            continue
        converter_fn = cvt._ONNX_NODE_REGISTRY[op_type]
        layer = converter_fn(node,graph,err)
        if type(layer)==tuple:
            for l in layer:
                layers.append(l)
        else:
            layers.append(layer)
        outs = node.outputs
        for out in outs:
            exist_edges.append(out)

    net = caffe_pb2.NetParameter()
    for id,layer in enumerate(layers):
        layers[id] = layer._to_proto()
    net.layer.extend(layers)

    with open(prototxt_save_path, 'w') as f:
        print(net,file=f)

    caffe.set_mode_cpu()
    deploy = prototxt_save_path
    net = caffe.Net(deploy,
                    caffe.TEST)

    for id, node in enumerate(graph.nodes):
        node_name = node.name
        op_type = node.op_type
        inputs = node.inputs
        inputs_tensor = node.input_tensors
        input_non_exist_flag = False
        if op_type not in wlr._ONNX_NODE_REGISTRY:
            err.unsupported_op(node)
            continue
        converter_fn = wlr._ONNX_NODE_REGISTRY[op_type]
        converter_fn(net, node, graph, err)

    net.save(caffe_model_save_path)
    return net

def getGraph(onnx_path):
    model = onnx.load(onnx_path)
    opset_version = model.opset_import[0].version  # 获取 opset version ,不同的 opset version 下 onnx的 op解析方式不同
    model = shape_inference.infer_shapes(model)
    model_graph = model.graph
    graph = Graph.from_onnx(model_graph)
    graph = graph.transformed(transformers)
    graph.channel_dims = {}

    return graph, opset_version


def export(model, onnx_path, dummy_input=(1,3,224,416), to_caffe=True):
    # Load model
    import torch
    model = model.to('cpu').eval()
    img = torch.zeros(*dummy_input).to('cpu')
    for _ in range(2):
        model(img)  # dry runs

    # ONNX model export
    prefix = 'ONNX:'
    try:
        import onnx
        print(f'\n{prefix} starting export with onnx {onnx.__version__}...')
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            verbose=True,
            keep_initializers_as_inputs=False,
            do_constant_folding=True,
            opset_version=9,
            input_names=['images'],
            output_names=['output'],
        )
        # Checks
        model_onnx = onnx.load(onnx_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        print(f'{prefix} export success, saved as {onnx_path}')
    except Exception as e:
        print(f'{prefix} export failure: {e}')

    # Convert to caffe
    if to_caffe:
        prototxt_path = onnx_path.replace('.onnx', '.prototxt')
        caffemodel_path = onnx_path.replace('.onnx', '.caffemodel')
        graph = getGraph(onnx_path)
        graph, opset_version = getGraph(onnx_path)
        convertToCaffe(graph, opset_version, prototxt_path, caffemodel_path)
        compareOnnxAndCaffe(onnx_path,prototxt_path,caffemodel_path)

if __name__ == "__main__":
    onnx_path = sys.argv[1]
    prototxt_path = sys.argv[2]
    caffemodel_path = sys.argv[3]
    graph = getGraph(onnx_path)
    graph, opset_version = getGraph(onnx_path)
    convertToCaffe(graph, opset_version, prototxt_path, caffemodel_path)
    compareOnnxAndCaffe(onnx_path,prototxt_path,caffemodel_path)

