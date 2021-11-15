import torch
import os


def to_onnx(
    model,
    onnx_path,
    dummy_input=(1, 3, 224, 416),
    input_names=["input"],
    output_names=["output.1", "output.2", "output.3"],
):
    # create dir
    dir = '/'.join(onnx_path.split('/')[:-1])
    if not os.path.exists(dir):
        os.makedirs(dir)

    # load model
    device = "cpu"
    model = model.to(device).eval()
    dummy_input = torch.rand(*dummy_input).to(device)
    model.forward(dummy_input)  # dry run

    # export ONNX model
    try:
        import onnx
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            opset_version=12,
            input_names=input_names,
            output_names=output_names,
        )
        # check
        model_onnx = onnx.load(onnx_path)
        onnx.checker.check_model(model_onnx)
    except Exception as e:
        raise e
    return onnx_path
