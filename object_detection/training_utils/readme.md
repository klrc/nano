## Augmentation Settings

### mosaic
![](https://pic3.zhimg.com/v2-aa1f61d2f1b4071c32c2670ef8257b2a_r.jpg)

### mixup
![](https://pic2.zhimg.com/80/v2-2458b46f8d9ae126b1b23f36345a0cbd_1440w.jpg)
```
Notice: only used for large models in u-yolov5 repository.
```

### degrees, translate, scale, shear, perspective (random affine)
![](https://pic1.zhimg.com/v2-86605e432a6801dff735ee05bfc5e360_r.jpg)
```
degrees: image rotation (+/- deg)
translate: image translation (+/- fraction)
scale: image scale (+/- gain)
shear: image shear (+/- deg)
perspective: image perspective (+/- fraction), range 0-0.001
```
### copy_paste
![](https://pic3.zhimg.com/v2-87f5aa536473dd6e109fc07360fe28ee_r.jpg)

### augment_hsv
![](https://pic4.zhimg.com/80/v2-aaec0670572fd5921e901685c4ea7e93_1440w.jpg)

### fake_osd
`training_utils.yolov5_dataset_loader_pack.augmentations.py:290`:
| ![](docs/fake_osd_off.png) | ![](docs/fake_osd_on.png) |
| -- | -- |
| fake_osd off | fake_osd on |



### fake_darkness
`training_utils.yolov5_dataset_loader_pack.augmentations.py:34`:

```python
    if fake_darkness:
        T.extend(
            [
                A.RandomGamma(p=0.01),
                A.ColorJitter(brightness=(0.1, 0.5), p=0.5),
                A.ImageCompression(quality_lower=75, p=0.5),
            ]
        )
```

| ![](docs/fake_darkness_off.png) | ![](docs/fake_darkness_on.png) |
| -- | -- |
| fake_darkness off | fake_darkness on |