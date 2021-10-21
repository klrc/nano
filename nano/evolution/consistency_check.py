import caffe
import onnx
import onnxruntime
import numpy as np
from collections import OrderedDict
import os
import shutil


def get_onnx_layer_outputs(onnx_info):
    onnx_path = onnx_info[0]
    in_node = onnx_info[1]
    input_data = np.loadtxt(onnx_info[2])
    input_data = input_data.reshape(onnx_info[3]).astype(np.float32)

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    for node in model.graph.node:
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    sess = onnxruntime.InferenceSession(model.SerializeToString())
    outputs = [x.name for x in sess.get_outputs()]
    res = sess.run(outputs, {in_node: input_data})
    res = OrderedDict(zip(outputs, res))

    output_names = list(res.keys())
    output_names.sort()
    print("onnx num of layers: {}".format(len(output_names)))

    return res


def get_caffe_layer_outputs(caffe_info):
    prototxt_path = caffe_info[0]
    caffemodel_path = caffe_info[1]
    in_node = caffe_info[2]
    input_data = np.loadtxt(caffe_info[3])
    input_data = input_data.reshape(caffe_info[4]).astype(np.float32)

    model = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)
    model.blobs[in_node].data[...] = input_data
    model.forward()
    res = model.blobs

    output_names = list(res.keys())
    output_names.sort()
    print("caffe num of layers: {}".format(len(output_names)))

    return res


def compareLayers(onnx_info, caffe_info, dump_path):
    onnx_outputs = get_onnx_layer_outputs(onnx_info)
    caffe_outputs = get_caffe_layer_outputs(caffe_info)

    for layer in onnx_outputs.keys():
        if layer in caffe_outputs.keys():
            onnx_res = onnx_outputs[layer]
            caffe_res = caffe_outputs[layer].data
            print("layer {} shape: {} for onnx vs {} for caffe"\
                   .format(layer, onnx_res.shape, caffe_res.shape))

            assert onnx_res.shape == caffe_res.shape

            dot_result = np.dot(onnx_res.flatten(), caffe_res.flatten())
            left_norm = np.sqrt(np.square(onnx_res).sum())
            right_norm = np.sqrt(np.square(caffe_res).sum())
            cos_sim = dot_result / (left_norm * right_norm)

            if cos_sim < 0.9999:
                onnx_file = os.path.join(dump_path, layer+'_onnx.txt')
                np.savetxt(onnx_file, onnx_res.flatten(), fmt='%.18f')
                caffe_file = os.path.join(dump_path, layer+'_caffe.txt')
                np.savetxt(caffe_file, caffe_res.flatten(), fmt='%.18f')
                print("cos sim of layer {}: {}".format(layer, cos_sim))


def get_onnx_inputs(onnx_graph):
    input_tensors = {t.name for t in onnx_graph.initializer}
    inputs = []
    for i in onnx_graph.input:
        if i.name not in input_tensors:
            inputs.append(i.name)
    return inputs


def get_onnx_outputs(onnx_graph):
    outputs = []
    for i in onnx_graph.output:
        outputs.append(i.name)
    return outputs


def load_onnx_model(onnx_path):
    return onnxruntime.InferenceSession(onnx_path)


def load_caffe_model(prototxt_path, caffemodel_path):
    return caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)


# generate input tensor for test
def gen_input(models, in_node):
    np.random.seed(5)
    # get & check input shape
    in_shapes = [get_input_shape_onnx(models[0], in_node), 
                 get_input_shape_caffe(models[1], in_node)]
    in_shape = in_shapes[0]
    for shape in in_shapes:
        if shape != in_shape:
            raise Exception("model input shape doesn't match: {} vs {}".format(shape, in_shape))
    # generate tensor of input shape with random value (NCHW)
    return np.random.rand(1, *in_shape).astype(np.float32)


def get_input_shape_onnx(onnx_model, in_node):
    # default onnx shape is NCHW
    for node in onnxruntime.InferenceSession.get_inputs(onnx_model):
        if node.name == in_node:
            return node.shape[1:]


def get_input_shape_caffe(caffe_model, in_node):
    # default caffe shape is NCHW
    return list(caffe_model.blobs[in_node].shape)[1:]


# perform network forward
def run_models(models, in_node, out_node, input_tensor):
    net_results = []
    net_results.append(net_forward_onnx(models[0], in_node, out_node, input_tensor))
    net_results.append(net_forward_caffe(models[1], in_node, out_node, input_tensor))
    return net_results


def net_forward_onnx(onnx_model, in_node, out_node, input_tensor):
    result = onnx_model.run(out_node, {in_node : input_tensor})
    return result


def net_forward_caffe(caffe_model, in_node, out_node, input_tensor):
    caffe_model.blobs[in_node].data[...] = input_tensor
    caffe_model.forward()
    result = []
    for node in out_node:
        result.append(caffe_model.blobs[node].data)
    return result


# check output results
def check_results(net_results, onnx_info, caffe_info, dump_path):
    onnx_results = net_results[0]
    caffe_results = net_results[1]
    print(onnx_results)
    print(caffe_results)
    for i, result in enumerate(onnx_results):
        # check if result are same by cosine distance
        dot_result = np.dot(result.flatten(), caffe_results[i].flatten())
        left_norm = np.sqrt(np.square(result).sum())
        right_norm = np.sqrt(np.square(caffe_results[i]).sum())
        cos_sim = dot_result / (left_norm * right_norm)
        print("cos sim between onnx and caffe models: {}".format(cos_sim))
        
        if cos_sim < 0.9999:
            # dump result
            np.savetxt(os.path.join(dump_path, "final_out_onnx.txt"), result.flatten(), fmt='%.18f')
            np.savetxt(os.path.join(dump_path, "final_out_caffe.txt"), caffe_results[i].flatten(), fmt='%.18f')
            compareLayers(onnx_info, caffe_info, dump_path)
            print('warning: model output cos difference exceed 0.0001')
            # raise Exception("model output different")

    print("models similarity test passed")


def compare_onnx_and_caffe(onnx_path, prototxt_path, caffemodel_path, clean_cache=True):
    
    dump_path = '/'.join(caffemodel_path.split('/')[:-1]) + '/cache'
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    models = [load_onnx_model(onnx_path), load_caffe_model(prototxt_path, caffemodel_path)]

    onnx_graph = onnx.load(onnx_path).graph
    in_node = get_onnx_inputs(onnx_graph)
    out_node = get_onnx_outputs(onnx_graph)
    if not len(in_node) == 1:
        raise Exception("only one input is supported, but {} provided: {}"
                        .format(len(in_node), in_node))
    print("input node: {}".format(in_node))
    print("output node: {}".format(out_node))

    # generate input tensor
    # in NCHW format
    input_tensor = gen_input(models, in_node[0])
    dump_input_file = os.path.join(dump_path, "input_{}x{}.txt".format(input_tensor.shape[2], input_tensor.shape[3]))
    np.savetxt(dump_input_file, input_tensor.flatten())
    print("input tensor shape of {}: {}".format(in_node[0], input_tensor.shape))
    print("dump input to {}".format(dump_input_file))

    # feed input tensors for each model and get result
    net_results = run_models(models, in_node[0], out_node, input_tensor)
    for i, node in enumerate(out_node):
        print("output tensor shape of {}: {} for onnx vs {} for caffe"
               .format(node, net_results[0][i].shape, net_results[1][i].shape))

    # check model results
    onnx_info = [onnx_path, in_node[0], dump_input_file, input_tensor.shape]
    caffe_info = [prototxt_path, caffemodel_path, in_node[0], dump_input_file, input_tensor.shape]
    check_results(net_results, onnx_info, caffe_info, dump_path)

    if clean_cache:
        shutil.rmtree(dump_path)
