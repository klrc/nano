from .onnx_utils import to_onnx
from .caffe_utils import onnx_to_caffe


def freeze(
    model,
    onnx_path,
    dummy_input=(1, 3, 224, 416),
    input_names=["input"],
    output_names=["output.1", "output.2", "output.3"],
    to_caffe=True,
    check_consistency=True,
):  
    print(f'build {onnx_path}')
    to_onnx(model, onnx_path, dummy_input, input_names, output_names)
    if to_caffe:
        print("build", onnx_path.replace(".onnx", ".caffemodel"))
        print("build", onnx_path.replace(".onnx", ".prototxt"))
        onnx_to_caffe(onnx_path, check_consistency)
