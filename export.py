
# import nano
# import torch

# model = nano.models.yolov5_mobilenetv3_l(num_classes=6)
# model.load_state_dict(torch.load('runs/train/exp36/weights/best.pt', map_location='cpu')['state_dict'])
# detector = nano.datasets.object_detection.detector

# nano.onnx.freeze(model, 'runs/build/YOLOV1000-MKII.onnx', opset=9, to_caffe=True)


import nano.onnx.test as test

test()
