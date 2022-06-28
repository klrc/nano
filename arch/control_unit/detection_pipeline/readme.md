## Get Started

### full dataset check
```python
import torch

if __name__ == "__main__":
    from arch.control_unit.detection_pipeline import full_dataset_test, coco_class_names
    from arch.model.object_detection.yolov5 import yolov5n

    model = yolov5n(80, weights=".yolov5_checkpoints/yolov5n_sd.pt", activation=torch.nn.SiLU).eval()

    # Full dataset test
    full_dataset_test(model, "../datasets/FH_test_walking_day_night/test_frames/images", class_names=coco_class_names)
```

### test on video
```python
import torch

if __name__ == "__main__":
    from arch.control_unit.detection_pipeline import video_test, coco_class_names
    from arch.model.object_detection.yolov5 import yolov5n
    from arch.io.video_loader import YUV420_VID_LOADER

    model = yolov5n(80, weights=".yolov5_checkpoints/yolov5n_sd.pt", activation=torch.nn.SiLU).eval()

    # Video test
    loader = YUV420_VID_LOADER("/Users/sh/Downloads/1280x720_2.yuv", (720, 1280))
    video_test(model, loader, class_names=coco_class_names, fps=16)
```

### test on front camera
```python
import torch

if __name__ == "__main__":
    from arch.control_unit.detection_pipeline import camera_test, coco_class_names
    from arch.model.object_detection.yolov5 import yolov5n

    model = yolov5n(80, weights=".yolov5_checkpoints/yolov5n_sd.pt", activation=torch.nn.SiLU).eval()

    # Camera test
    camera_test(model, class_names=coco_class_names, inf_size=480)
```