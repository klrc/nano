from object_detection.training_utils.default_settings import DefaultSettings
from object_detection.yolov5_ultralytics import yolov5n
from test_utils.detection_test import camera_test

model = yolov5n(80, weights='runs/yolov5n_hand_only.0/fuse.pt').eval()
camera_test(model, class_names=DefaultSettings.names, inf_size=480)
