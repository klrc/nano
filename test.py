import torch.nn as nn
from object_detection.training_utils.default_settings import DefaultSettings

from test_utils.detection_test import full_dataset_test, video_test, camera_test
from test_utils.video_loader import YUV420_VID_LOADER

if __name__ == "__main__":
    import object_detection.yolov5_ultralytics.model as yolo_u

    yolo_u.ACTIVATION = nn.SiLU
    model = yolo_u.yolov5n(80, weights="yolov5n_sd.pt").eval()

    # Full dataset test
    # full_dataset_test(model, "../datasets/FH_test_walking_day_night/test_frames/images", class_names=DefaultSettings.names)

    # # Video test
    # loader = YUV420_VID_LOADER("/Users/sh/Downloads/1280x720_2.yuv", (720, 1280))
    # video_test(model, loader, class_names=DefaultSettings.names, fps=16)

    # Camera test
    camera_test(model, class_names=DefaultSettings.names, inf_size=480)
