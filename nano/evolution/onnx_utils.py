import torch
from torch import onnx


def to_onnx(
    model,
    onnx_path,
    dummy_input=(1, 3, 224, 416),
    input_names=["input"],
    output_names=["output.1", "output.2", "output.3"],
):
    # load model
    device = "cpu"
    model = model.to(device).eval()
    if hasattr(model, "dsp"):
        model = model.dsp()
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
