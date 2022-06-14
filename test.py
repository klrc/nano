import cv2
import torch.nn as nn
from object_detection.training_utils.default_settings import DefaultSettings

from test_utils.detection_test import detect

if __name__ == "__main__":
    image = cv2.imread("000000011197.jpg")
    import object_detection.yolov5_ultralytics.model as yolo_u

    yolo_u.ACTIVATION = nn.SiLU
    model = yolo_u.yolov5s(80, weights="yolov5s_sd.pt").eval()
    detect(model, image, class_names=DefaultSettings.names)
