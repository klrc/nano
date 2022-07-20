### Get Started

to validate a model, you need to set:
- a **4C** model
- a dataloader (Iterable, with `__len__()` function)
- class_names (as a list)

---
Example Code:

```python
from arch.model import yolov5n_4c
from arch.controller.detection_validation_4c import detection_validation
from arch.data.detection_annotation import coco_meta_data, Yolo5Dataloader
import torch

if __name__ == "__main__":
    model = yolov5n_4c(weight=".yolov5_checkpoints/yolov5n_sd.pt", pretrained=True).replace_activation(torch.nn.SiLU)
    val_loader = Yolo5Dataloader('../datasets/coco128/images/train2017', 640, 4)

    detection_validation(model, val_loader, coco_meta_data.names)
```