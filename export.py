import nano
from nano.evolution import freeze


model = nano.models.yolox_cspm()
# model.load_state_dict(torch.load('runs/train/exp36/weights/best.pt', map_location='cpu')['state_dict'])

freeze(model, "runs/build/YOLOV1000-MKII/YOLOV1000-MKII.onnx", to_caffe=True)
