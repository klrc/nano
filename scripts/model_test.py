import torch
import nano
import yaml
from nano.detection import CaffeWrapper, evaluator


model = nano.models.esnet_cspp_yolov5_s4__seblock_canceled()
model.load_state_dict(torch.load("runs/train/exp194/weights/best.pt", map_location="cpu")["state_dict"])
model.names = {0: "person", 1: "bike", 2: "car"}

evaluator = nano.detection.evaluator
evaluator.run(
    model,
    # data="configs/coco-val.yaml",
    data="configs/coc-x.yaml",
    batch_size=32,
    imgsz=416,
    device="cpu",
)


# [sh@sh-20hhcto1ww nano]$  /usr/bin/env /usr/bin/python /home/sh/.vscode-server/extensions/ms-python.python-2021.12.1559732655/pythonFiles/lib/python/debugpy/launcher 46343 -- /home/sh/Projects/klrc/nano/scripts/model_test.py 
# YOLOv5 🚀 dfe4375 torch 1.9.0+cu102 CPU

# val: Scanning '../datasets/coc-x/val.cache' images and labels... 6452 found, 0 missing, 3723 empty, 0 corrupted: 100%|█| 6452/
#                Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|█| 404/404 [09:47<00:00,  1.45s/i
#                  all       6452       6391      0.767      0.625      0.701      0.403
#                    0       6452       4372      0.715      0.668      0.715       0.39
#                    1       6452        714      0.865      0.535      0.686      0.364
#                    2       6452       1305       0.72      0.672      0.704      0.457
# Speed: 0.5ms pre-process, 88.8ms inference, 1.1ms NMS per image at shape (32, 3, 416, 416)
# [sh@sh-20hhcto1ww nano]$  cd /home/sh/Projects/klrc/nano ; /usr/bin/env /usr/bin/python /home/sh/.vscode-server/extensions/ms-python.python-2021.12.1559732655/pythonFiles/lib/python/debugpy/launcher 32845 -- /home/sh/Projects/klrc/nano/scripts/model_test.py 
# YOLOv5 🚀 dfe4375 torch 1.9.0+cu102 CPU

# val: Scanning '../datasets/coc-x/val.cache' images and labels... 6452 found, 0 missing, 3723 empty, 0 corrupted: 100%|█| 6452/
#                Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|█| 404/404 [10:22<00:00,  1.54s/i
#                  all       6452       6391      0.186     0.0706     0.0365      0.015
#                    0       6452       4372      0.148     0.0555      0.024     0.0106
#                    1       6452        714      0.216     0.0098    0.00658    0.00243
#                    2       6452       1305      0.194      0.146     0.0788     0.0319
# Speed: 0.5ms pre-process, 94.7ms inference, 0.8ms NMS per image at shape (32, 3, 416, 416)
# [sh@sh-20hhcto1ww nano]$  cd /home/sh/Projects/klrc/nano ; /usr/bin/env /usr/bin/python /home/sh/.vscode-server/extensions/ms-python.python-2021.12.1559732655/pythonFiles/lib/python/debugpy/launcher 38547 -- /home/sh/Projects/klrc/nano/scripts/model_test.py 
# YOLOv5 🚀 ec922a9 torch 1.9.0+cu102 CPU

# val: Scanning '../datasets/coc-x/val.cache' images and labels... 6452 found, 0 missing, 3723 empty, 0 corrupted: 100%|█| 6452/
#                Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|█| 404/404 [12:26<00:00,  1.85s/i
#                  all       6452       6391      0.758      0.627      0.697      0.401
#                    0       6452       4372      0.708       0.67       0.71      0.388
#                    1       6452        714      0.865      0.535      0.685      0.363
#                    2       6452       1305        0.7      0.675      0.697      0.452
# Speed: 0.7ms pre-process, 112.0ms inference, 2.1ms NMS per image at shape (32, 3, 416, 416)