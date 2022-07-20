## Get Started
### Quick Training
to start a quick training, you need:
- a **4C** pytorch model (`torch.nn.Module`)
- train_loader & val_loader
- loss function

```python
from arch.model import yolov5n_4c
from arch.controller.detection_training_4c import DetectionTraining4C
from arch.data.detection_annotation import Yolo5Dataloader
import torch

if __name__ == "__main__":
    model = yolov5n_4c(weight=".yolov5_checkpoints/yolov5n_sd.pt", pretrained=True).replace_activation(torch.nn.SiLU)
    train_loader = Yolo5Dataloader('../datasets/coco128/images/train2017', 640, 2)
    val_loader = Yolo5Dataloader('../datasets/coco128/images/train2017', 640, 2)

    DT = DetectionTraining4C()
    DT.train(model, train_loader, val_loader, model.head.loss)
```
### Custom Settings
to add custom settings, change the attributes of `DetectionTraining41C` instance before training, settings will be recorded into `frozen.core` file during training.

```python
from arch.model import yolov5n_4c
from arch.controller.detection_training_4c import DetectionTraining4C
from arch.data.detection_annotation import Yolo5Dataloader
import torch

if __name__ == "__main__":
    model = yolov5n_4c(weight=".yolov5_checkpoints/yolov5n_sd.pt", pretrained=True).replace_activation(torch.nn.SiLU)
    train_loader = Yolo5Dataloader('../datasets/coco128/images/train2017', 640, 2)
    val_loader = Yolo5Dataloader('../datasets/coco128/images/train2017', 640, 2)

    DT = DetectionTraining4C()
    DT.lrf = 0.005
    DT.lr0 = 0.001
    DT.device = 0
    DT.save_dir = "runs/my_custom_model"
    DT.train(model, train_loader, val_loader, model.head.loss)
```

### Resume Training
the `frozen.core` file contains all the parameters for training, you can kill & resume at any time with exactly the same result.

```python
from arch.model import yolov5n_4c
from arch.controller.detection_training_4c import DetectionTraining4C
from arch.data.detection_annotation import Yolo5Dataloader
import torch

if __name__ == "__main__":
    model = yolov5n(weight=".yolov5_checkpoints/yolov5n_sd.pt", pretrained=True).replace_activation(torch.nn.SiLU)
    train_loader = Yolo5Dataloader('../datasets/coco128/images/train2017', 640, 2)
    val_loader = Yolo5Dataloader('../datasets/coco128/images/train2017', 640, 2)

    DT = DetectionTraining4C()
    DT.resume('runs/untitled_model.2/frozen.core', model, train_loader, val_loader, model.head.loss)
```

