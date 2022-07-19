## Get Started

### quick test: pretrained yolov5n
```python
from arch.model import yolov5n
from arch.controller.detection_pipeline import DetectPipeline41C
from arch.data.video import LiveCamera
from arch.data.detection_annotation import coco_meta_data
import torch

if __name__ == "__main__":
    model = yolov5n(weight=".yolov5_checkpoints/yolov5n_sd.pt", pretrained=True)
    model.replace_activation(torch.nn.SiLU)

    data = LiveCamera(inf_size=416)
    det = DetectPipeline41C(model, class_names=coco_meta_data.names)

    det.stream(data, auto=True)
```

## What is 41C/4C ?
we mark `41C` as the original format of yolov5 output, as (XYWH, Conf, Score_0, Score_1, ..., Score_{n-1})
while `4C` means joint representation for yolov5 output, which removes the confidence branch. See [https://zhuanlan.zhihu.com/p/147691786](https://zhuanlan.zhihu.com/p/147691786)