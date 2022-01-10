
import os
import torch
from .. import xnnc_caffe
from torch.onnx import OperatorExportTypes


def make(model, model_name, cache_dir, output_names, dummy_input_shape=(1, 3, 224, 416)):
    # create dirs
    for _dir in [cache_dir]:
        if not os.path.exists(_dir):
            os.makedirs(_dir)
    # pt -> onnx
    onnx_path = f"{cache_dir}/{model_name}.onnx"
    torch.onnx.export(
        model,
        args=torch.rand(*dummy_input_shape),
        f=onnx_path,
        input_names=["input"],
        output_names=output_names,
        opset_version=12,
        operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
        enable_onnx_checker=True,
        do_constant_folding=False,
        keep_initializers_as_inputs=True,
        verbose=True,
    )
    # onnx -> caffe
    return xnnc_caffe.export(
        onnx_path,
    )
