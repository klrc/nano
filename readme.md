# NANO
轻量级目标检测项目代码仓库 （或许还有其他


## Architecture
- nano: main source
    - nano.detection: object detection related modules
    - nano.datasets: dataset loader scripts
    - nano.models: models (under development)
    - nano._utils: xnnc porting & log utils
- release: released models, each released model includes the following items:
    - *.pt
    - *.onnx (optional)
    - *.caffemodel
    - *-custom.caffemodel (optional)
    - *.prototxt
    - *-custom.prototxt
    - readme.txt (containing anchors,metrics and description)
- configs: including preset yaml configurations.
- ~~runs: running logs~~ (not sync to git)
- detect.py
- export.py
- train.py
- readme.md
- model_sheet.md

## Current TODO
- add wandb support
- add caffe evaluator / inference wrapper

## -

---

![](nano.jpg)
