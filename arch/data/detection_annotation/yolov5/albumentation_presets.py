import random
from loguru import logger
import numpy as np

from .utils import check_version, colorstr


class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self, fake_darkness):
        self.transform = None
        try:
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)  # version requirement

            T = [
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
            ]  # transforms
            if fake_darkness:
                T.extend(
                    [
                        A.RandomGamma(p=0.01),
                        A.ColorJitter(brightness=(0.15, 0.5), p=0.5),
                        A.ImageCompression(quality_lower=75, p=0.5),
                    ]
                )
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

            print(colorstr("albumentations: ") + ", ".join(f"{x}" for x in self.transform.transforms if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            logger.error(colorstr("albumentations: ") + f"{e}")

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new["image"], np.array([[c, *b] for c, b in zip(new["class_labels"], new["bboxes"])])
        return im, labels

