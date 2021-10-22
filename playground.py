import nano
from nano.evolution import to_onnx, onnx_to_caffe


model = nano.models.tmp_test_yolov5_cspm()

onnx_path = to_onnx(model, "runs/build/test.onnx")
onnx_to_caffe(onnx_path)
