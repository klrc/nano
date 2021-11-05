from caffe.proto import caffe_pb2
from caffe import params as P
import six
import math


def param_name_dict():
    """Find out the correspondence between layer names and parameter names."""

    layer = caffe_pb2.LayerParameter()
    # get all parameter names (typically underscore case) and corresponding
    # type names (typically camel case), which contain the layer names
    # (note that not all parameters correspond to layers, but we'll ignore that)
    param_names = [f.name for f in layer.DESCRIPTOR.fields if f.name.endswith("_param")]
    param_type_names = [type(getattr(layer, s)).__name__ for s in param_names]
    # strip the final '_param' or 'Parameter'
    param_names = [s[: -len("_param")] for s in param_names]
    param_type_names = [s[: -len("Parameter")] for s in param_type_names]
    return dict(zip(param_type_names, param_names))


def assign_proto(proto, name, val):
    """Assign a Python object to a protobuf message, based on the Python
    type (in recursive fashion). Lists become repeated fields/messages, dicts
    become messages, and other types are assigned directly. For convenience,
    repeated fields whose values are not lists are converted to single-element
    lists; e.g., `my_repeated_int_field=3` is converted to
    `my_repeated_int_field=[3]`."""

    is_repeated_field = hasattr(getattr(proto, name), "extend")
    if is_repeated_field and not isinstance(val, list):
        val = [val]
    if isinstance(val, list):
        if isinstance(val[0], dict):
            for item in val:
                proto_item = getattr(proto, name).add()
                for k, v in six.iteritems(item):
                    assign_proto(proto_item, k, v)
        else:
            getattr(proto, name).extend(val)
    elif isinstance(val, dict):
        for k, v in six.iteritems(val):
            assign_proto(getattr(proto, name), k, v)
    else:
        setattr(proto, name, val)


class Function(object):
    """A Function specifies a layer, its parameters, and its inputs (which
    are Tops from other layers)."""

    def __init__(self, type_name, layer_name, inputs, outputs, **params):
        self.type_name = type_name
        self.inputs = inputs
        self.outputs = outputs
        self.params = params
        self.layer_name = layer_name
        self.ntop = self.params.get("ntop", 1)
        # use del to make sure kwargs are not double-processed as layer params
        if "ntop" in self.params:
            del self.params["ntop"]
        self.in_place = self.params.get("in_place", False)
        if "in_place" in self.params:
            del self.params["in_place"]
        # self.tops = tuple(Top(self, n) for n in range(self.ntop))l

    def _get_name(self, names, autonames):
        if self not in names and self.ntop > 0:
            names[self] = self._get_top_name(self.tops[0], names, autonames)
        elif self not in names:
            autonames[self.type_name] += 1
            names[self] = self.type_name + str(autonames[self.type_name])
        return names[self]

    def _get_top_name(self, top, names, autonames):
        if top not in names:
            autonames[top.fn.type_name] += 1
            names[top] = top.fn.type_name + str(autonames[top.fn.type_name])
        return names[top]

    def _to_proto(self):
        bottom_names = []
        for inp in self.inputs:
            # inp._to_proto(layers, names, autonames)
            bottom_names.append(inp)
        layer = caffe_pb2.LayerParameter()
        layer.type = self.type_name
        layer.bottom.extend(bottom_names)

        if self.in_place:
            layer.top.extend(layer.bottom)
        else:
            for top in self.outputs:
                layer.top.append(top)
        layer.name = self.layer_name
        # print(self.type_name + "...")
        for k, v in six.iteritems(self.params):
            # special case to handle generic *params
            # print("generating "+k+"...")

            if k.endswith("param"):
                assign_proto(layer, k, v)
            else:
                try:
                    assign_proto(getattr(layer, _param_names[self.type_name] + "_param"), k, v)
                except (AttributeError, KeyError):
                    assign_proto(layer, k, v)

        return layer


class Layers(object):
    """A Layers object is a pseudo-module which generates functions that specify
    layers; e.g., Layers().Convolution(bottom, kernel_size=3) will produce a Top
    specifying a 3x3 convolution applied to bottom."""

    def __getattr__(self, name):
        def layer_fn(*args, **kwargs):
            fn = Function(name, args, kwargs)
            return fn

        return layer_fn


_param_names = param_name_dict()


def conv2d(node_name, input_names, output_names, out_channels, kernel_size, stride, padding, groups, dilation, bias):
    # register weight shape
    assert padding[0] == padding[2] and padding[1] == padding[3]
    return Function(
        "Convolution",
        node_name,
        input_names,
        output_names,
        kernel_h=kernel_size[0],
        kernel_w=kernel_size[1],
        stride_h=stride[0],
        stride_w=stride[1],
        group=groups,
        pad_h=padding[0],
        pad_w=padding[1],
        num_output=out_channels,
        dilation=dilation[0],
        bias_term=bias,
    )


def relu(node_name, input_names, output_names, inplace=False):
    return Function("ReLU", node_name, input_names, output_names, in_place=inplace)


def concat(node_name, input_names, output_names, axis):
    return Function("Concat", node_name, input_names, output_names, axis=axis)


def maxpool2d(node_name, input_names, output_names, kernel_size, stride, padding):
    assert padding[0] == padding[2] and padding[1] == padding[3]
    return Function(
        "Pooling",
        node_name,
        input_names,
        output_names,
        pooling_param=dict(
            pool=P.Pooling.MAX,
            kernel_h=kernel_size[0],
            kernel_w=kernel_size[1],
            stride_h=stride[0],
            stride_w=stride[1],
            pad_h=padding[0],
            pad_w=padding[1],
        ),
    )


def avgpool2d(node_name, input_names, output_names, kernel_size, stride, padding):
    assert padding[0] == padding[2] and padding[1] == padding[3]
    return Function(
        "Pooling",
        node_name,
        input_names,
        output_names,
        pooling_param=dict(
            pool=P.Pooling.AVE,
            kernel_h=kernel_size[0],
            kernel_w=kernel_size[1],
            stride_h=stride[0],
            stride_w=stride[1],
            pad_h=padding[0],
            pad_w=padding[1],
        ),
    )


def add(node_name, input_names, output_names):
    return Function(
        "Eltwise",
        node_name,
        input_names,
        output_names,
        operation=P.Eltwise.SUM,
    )


def flatten(node_name, input_names, output_names):
    return Function(
        "Flatten",
        node_name,
        input_names,
        output_names,
    )


def bias(node_name, input_names, output_names, axis):
    return Function(
        "Bias",
        node_name,
        input_names,
        output_names,
        axis=axis,
    )


def upsample_bilinear2d(node_name, input_names, output_names, in_channels, scale_factor):
    kernel_h = int(2 * scale_factor[0] - scale_factor[0] % 2)
    kernel_w = int(2 * scale_factor[1] - scale_factor[1] % 2)
    stride_h = int(scale_factor[0])
    stride_w = int(scale_factor[1])
    pad_h = int(math.ceil((scale_factor[0]-1)/2.))
    pad_w = int(math.ceil((scale_factor[1]-1)/2.))
    return Function(
        "Deconvolution",
        node_name,
        input_names,
        output_names,
        convolution_param=dict(
            num_output=in_channels,
            kernel_h=kernel_h,
            kernel_w=kernel_w,
            stride_h=stride_h,
            stride_w=stride_w,
            pad_h=pad_h,
            pad_w=pad_w,
            group=in_channels,
            bias_term=False,
            weight_filler=dict(type="bilinear"),
        ),
    )


def input_node(node_name, output_names, output_shape):
    input_names = []
    return Function("Input", node_name, input_names, output_names, input_param=dict(shape=dict(dim=output_shape)))
