from caffe.proto import caffe_pb2
from google.protobuf import text_format


def read_proto(path):
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(path).read(), net)
    return net


def convert_prototxt(net, output_names):
    layers = net.layer
    shuffle_cid = 1
    for i, layer in enumerate(layers):
        # check detection pattern
        if layer.top in output_names:
            layers.pop(i)  # last transpose
            layers.pop(i-1)  # last reshape
        # check channel shuffle pattern
        # reshape - transpose - reshape - split
        elif i >= 2 and layers[i].type == 'Reshape'\
                and layers[i-1].type == 'Permute'\
                and layers[i-2].type == 'Reshape':
            # create customized layer
            xnnc_layer = caffe_pb2.LayerParameter()
            xnnc_layer.name = f"shuffle_{shuffle_cid}"
            shuffle_cid += 1
            xnnc_layer.type = "CppCustom"
            xnnc_layer.bottom.append(layers[i-2].bottom[0])
            xnnc_layer.top.append(layers[i].top[0])
            assert int(xnnc_layer.top[0]) - int(xnnc_layer.bottom[0]) == 5
            # pop original channel shuffle layers
            layers.pop(i)
            layers.pop(i-1)
            layers.pop(i-2)
            # add custom layers
            layers.insert(i-2, xnnc_layer)
    return net


def convert_to_xnnc(proto_path, output_names):
    print('converting to xnnc standarlized caffe.. ', end='')
    net = read_proto(proto_path)
    with open(proto_path, 'w') as f:
        f.write(str(convert_prototxt(net, output_names)))

    with open(proto_path, 'r') as f:
        lines = f.readlines()

    xnnc_channel_shuffle_params = \
        """  cpp_custom_param {
        module: "XnncShuffleChannel"
        param_map_str: "group:2 "
    }"""
    frank_slice_params = \
        """  cpp_custom_param {
        module: "XnncSliceLayer"
        param_map_str: "axis:${1} slicepoint:${2}"
    }"""

    new_lines = []
    for i, line in enumerate(lines):
        if i >= 4 and 'type: "CppCustom"' in lines[i-3] and 'shuffle' in lines[i-4]:
            for _l in xnnc_channel_shuffle_params.split('\n'):
                new_lines.append(_l+'\n')
        if i >= 8 and 'type: "Slice"' in lines[i-8]:
            new_lines.pop()
            slice_point = str(int(new_lines.pop().split(':')[-1]))
            slice_dim = str(int(new_lines.pop().split(':')[-1]))
            new_lines.pop()
            slice_params = frank_slice_params.replace('${1}', slice_dim).replace('${2}', slice_point)
            for _l in slice_params.split('\n'):
                new_lines.append(_l+'\n')
        new_lines.append(line)

    new_lines = [('  type: "CppCustom"\n' if 'type: "Slice"' in line else line) for line in new_lines]

    with open(proto_path, 'w') as f:
        for line in new_lines:
            f.write(line)
    
    print('done :)')
