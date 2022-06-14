import cv2
import torch.nn as nn

from test_utils.detection_test import detect
import torch

if __name__ == "__main__":
    image = cv2.imread("000000011197.jpg")
    import object_detection.yolov5_ultralytics.model as yolo_u

    yolo_u.ACTIVATION = nn.SiLU
    model = yolo_u.yolov5s(80, weights="yolov5s_sd.pt").eval()

    # torch.onnx.export(
    #     model,
    #     args=torch.rand(1, 3, 384, 640),
    #     f="test.onnx",
    #     input_names=["input"],
    #     opset_version=12,
    # )

    detect(model, image)
