import random
import numpy as np


class RandomFlip:
    def __init__(self, ud, lr):
        self.ud = ud
        self.lr = lr

    def __call__(self, image, labels_xywh):
        labels = labels_xywh
        nl = len(labels)
        # Flip up-down
        if random.random() < self.ud:
            image = np.flipud(image)
            if nl:
                labels[:, 2] = 1 - labels[:, 2]

        # Flip left-right
        if random.random() < self.lr:
            image = np.fliplr(image)
            if nl:
                labels[:, 1] = 1 - labels[:, 1]

        return image, labels
