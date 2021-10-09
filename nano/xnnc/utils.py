from caffe.proto import caffe_pb2
from google.protobuf import text_format

def read_proto(path):
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(path).read(), net)
    layers = net.layer
    return layers

def convert_to_xnnc(layers, output_names=['696', '676', 'output']):
    shuffle_cid = 1
    stack = []
    for layer in layers:
        # add layer
        stack.append(layer)
        # check detection pattern
        if layer.top in output_names:
            stack.pop()  # last transpose
            stack.pop()  # last reshape
        # check channel shuffle pattern
        # reshape - transpose - reshape - split
        if len(stack) >= 3 \
            and stack[-3].type == 'reshape'\
            and stack[-2].type == 'transpose'\
            and stack[-1].type == 'reshape':
            # create customized layer
            xnnc_layer = caffe_pb2.LayerParameter()
            xnnc_layer.name = "shuffle_" + shuffle_cid
            shuffle_cid += 1
            xnnc_layer.type = "CppCustom"
            xnnc_layer.bottom = stack[-3].bottom
            xnnc_layer.top = stack[-1].top
            assert xnnc_layer.top - xnnc_layer.bottom == 5
            xnnc_layer.cpp_custom_param = {
                'module': "XnncShuffleChannel",
                'param_map_str': "group:2 ",
            }
            # pop original channel shuffle layers
            for _ in range(3):
                stack.pop()
            # add custom layers
            stack.append(xnnc_layer)
        return stack


proto_path_no_suffix = 'release/yolov5_shufflenet_1_5x@coco-s+animal/yolov5_shufflenet_1_5x'
layers = read_proto(proto_path_no_suffix + '.prototxt')
with open(proto_path_no_suffix + '_xnnc.prototxt','w') as f:
	f.write(str(convert_to_xnnc(layers)))