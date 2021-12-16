# NANO
轻量级目标检测项目代码仓库 （或许还有其他


## Architecture
- nano: main source
    - nano.detection: object detection related modules
    - nano.datasets: dataset loader scripts
    - nano.models: models (under development)
    - nano._utils: xnnc porting & log utils
- release: released models, each released model includes the following items:
    - readme.yaml (containing anchors,metrics and description)
    - *-custom.prototxt
    - *-custom.caffemodel (optional)
    - *.caffemodel
    - *.onnx (optional)
    - *.prototxt
    - *.pt
    - *.py definition
- configs: including preset yaml configurations.
- ~~runs: running logs~~ (not sync to git)
- detect.py
- export.py
- train.py
- test.py
- realtime_test.py
- readme.md
- model_sheet.md

## Current TODO
- adjust channel_shuffle exporting strategy
- train yolo_defense_es_96h_4x (with focal loss at stage2)
- train reid head

## -

---

![](nano.jpg)
