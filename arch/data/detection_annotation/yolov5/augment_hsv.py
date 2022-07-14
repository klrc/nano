import cv2
import numpy as np


class AugmentHSV:
    def __init__(self, hgain, sgain, vgain):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        pass

    def __call__(self, image, labels):
        hgain, sgain, vgain = self.hgain, self.sgain, self.vgain
        # HSV color-space augmentation
        if hgain or sgain or vgain:
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
            dtype = image.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=image)
        return image, labels
