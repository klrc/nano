import torch


def to_onnx(
    model, onnx_path, dummy_input=(1, 3, 224, 416), opset_version=9, device="cpu"
):
    # load model
    model = model.to(device).eval()
    if hasattr(model, "dsp"):
        model = model.dsp()
    dummy_input = torch.rand(*dummy_input).to(device)
    model.forward(dummy_input)  # dry run

    # export ONNX model
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        verbose=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=True,
        opset_version=
    )
