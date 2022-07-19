### Get Started

to validate a model, you need to set:
- a 41C model (such as yolov5)
- a dataloader (Iterable, with `__len__()` function)
- class_names (as a list)

---
Example Code:

```python
from arch.model import yolov5n
from arch.controller.detection_validation_41c import detection_validation
from arch.data.detection_annotation import coco_meta_data, Yolo5Dataloader
import torch

if __name__ == "__main__":
    model = yolov5n(weight=".yolov5_checkpoints/yolov5n_sd.pt", pretrained=True).replace_activation(torch.nn.SiLU)
    val_loader = Yolo5Dataloader('../datasets/coco128/images/train2017', 640, 4)

    detection_validation(model, val_loader, coco_meta_data.names)
```

Expected Output:
```
(dev) sh@Hans-MacBook-Pro nano %  cd /Users/sh/Projects/nano ; /usr/bin/env /opt/homebrew/anaconda3/envs/dev/bin
/python /Users/sh/.vscode/extensions/ms-python.python-2022.10.1/pythonFiles/lib/python/debugpy/adapter/../../deb
ugpy/launcher 53365 -- /Users/sh/Projects/nano/test.py 
loading weight from .yolov5_checkpoints/yolov5n_sd.pt
transferred 349/349 items
Scanning '../datasets/coco128/labels/train2017.cache' images and labels... 128 found, 0 missing, 2 empty, 0 corr
albumentations: Blur(always_apply=False, p=0.01, blur_limit=(3, 7)), MedianBlur(always_apply=False, p=0.01, blur_limit=(3, 7)), ToGray(always_apply=False, p=0.01), CLAHE(always_apply=False, p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95:   0%|          | 0/16 [0/opt/homebrew/anaconda3/envs/dev/lib/python3.9/site-packages/torch/functional.py:445: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /Users/distiller/project/conda/conda-bld/pytorch_1639180852547/work/aten/src/ATen/native/TensorShape.cpp:2157.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 16/16 [
                 all        128        870      0.667      0.402       0.49      0.312
              person        128        228      0.792      0.585      0.679      0.421
             bicycle        128          3          1      0.651      0.676      0.338
                 car        128         45      0.622      0.147      0.226      0.122
          motorcycle        128          5      0.577        0.6      0.721      0.525
            airplane        128          6          1      0.721       0.86      0.486
                 bus        128          5      0.514        0.6      0.611      0.469
               train        128          3      0.737      0.667      0.706      0.456
               truck        128         12      0.489      0.167      0.279      0.162
                boat        128          6      0.703      0.333      0.436      0.283
       traffic light        128         10      0.365        0.1      0.211      0.178
           stop sign        128          2      0.968        0.5      0.828      0.518
               bench        128          9      0.342      0.222      0.342      0.154
                bird        128         16       0.84      0.562      0.708      0.426
                 cat        128          4      0.635        0.5       0.77      0.482
                 dog        128          9      0.464      0.444      0.546      0.349
               horse        128          2      0.739          1      0.995      0.498
            elephant        128         17      0.968      0.882      0.891      0.503
                bear        128          1      0.518          1      0.995      0.796
               zebra        128          4      0.849          1      0.995      0.895
             giraffe        128          8      0.843      0.674       0.94      0.611
            backpack        128          5      0.729        0.4      0.414      0.206
            umbrella        128         18      0.742      0.333      0.422      0.164
             handbag        128         18          1          0    0.00306    0.00275
                 tie        128          6      0.669      0.333      0.483      0.313
            suitcase        128          4      0.552          1      0.849      0.642
             frisbee        128          5      0.749        0.8      0.769      0.591
           snowboard        128          7          1      0.401       0.64      0.413
         sports ball        128          5       0.72        0.4      0.508      0.346
                kite        128         10      0.497        0.2      0.204     0.0631
        baseball bat        128          4      0.288      0.216      0.108      0.041
      baseball glove        128          6      0.519      0.333      0.404      0.232
          skateboard        128          5      0.617        0.2      0.427      0.195
       tennis racket        128          7      0.547      0.429       0.38      0.317
              bottle        128         18      0.528       0.25       0.28      0.163
          wine glass        128         15      0.345     0.0667      0.211     0.0714
                 cup        128         33      0.992      0.182      0.293      0.154
                fork        128          4       0.43       0.25      0.281      0.128
               knife        128         16      0.816      0.555      0.627      0.234
               spoon        128         16          1       0.17      0.336      0.154
                bowl        128         28      0.717      0.429      0.572      0.367
              banana        128          1          0          0     0.0474     0.0364
            sandwich        128          2          0          0      0.171      0.154
              orange        128          4          1          0     0.0958     0.0383
            broccoli        128         11      0.526      0.182      0.181      0.155
              carrot        128         24      0.223     0.0417      0.217      0.114
             hot dog        128          2      0.517        0.5      0.745       0.55
               pizza        128          5      0.695        0.8      0.866      0.641
               donut        128         14      0.667      0.857       0.91      0.661
                cake        128          4      0.666       0.75      0.912      0.672
               chair        128         34      0.359      0.382      0.284      0.115
               couch        128          6      0.844      0.333      0.613      0.324
        potted plant        128         14          1      0.351       0.68      0.368
                 bed        128          3          1          0       0.23     0.0924
        dining table        128         12          1      0.393      0.538      0.382
              toilet        128          2      0.586        0.5      0.515      0.463
                  tv        128          2      0.627          1      0.995      0.798
              laptop        128          3          1          0      0.168      0.123
               mouse        128          2          1          0          0          0
              remote        128          8      0.643       0.25      0.315      0.144
          cell phone        128          8      0.636      0.125       0.24      0.102
           microwave        128          2      0.586        0.5      0.638      0.547
                oven        128          5      0.388        0.4      0.385       0.21
                sink        128          6       0.36      0.167      0.275      0.128
        refrigerator        128          5      0.707        0.4      0.484      0.304
                book        128         29      0.865      0.069       0.25     0.0893
               clock        128          8      0.947      0.875      0.887      0.584
                vase        128          2      0.314        0.5      0.275      0.215
            scissors        128          1          1          0          0          0
          teddy bear        128         21      0.782      0.286      0.473      0.245
          toothbrush        128          5      0.305        0.2      0.268      0.137
```