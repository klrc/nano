import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from .utils import tensor2im


class Canvas:
    def __init__(self, image=None, backend="cv2") -> None:
        # random color queue
        self.color_presets = {}
        if image is not None:
            self.load(image)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.3
        self.font_thickness = 1
        self.font_color = (0, 0, 0)
        self.backend = backend

    def load(self, image, clone=True):
        self.image = self.check_cv2(image, clone)

    @staticmethod
    def check_cv2(image, clone):
        if isinstance(image, torch.Tensor):
            image = tensor2im(image)
        elif clone:
            image = image.copy()
        assert isinstance(image, np.ndarray)
        return image

    @staticmethod
    def _safe_pt(pt):
        # forced int point coords for cv2 functions
        return [int(x) for x in pt]

    def color(self, id):
        # get color from color queue, select random color if not found
        if id not in self.color_presets:
            self.color_presets[id] = list(np.random.random(size=3) * 128 + 128)  # light color
        return self.color_presets[id]

    def merge_visible(self, layer, alpha):
        # merge layer with alpha
        alpha = float(alpha)
        self.image = cv2.addWeighted(self.image, 1 - alpha, layer, alpha, 0)

    def draw_point(self, pt1, thickness=3, alpha=1, color=None):
        # draw a single point on canvas
        layer = self.image.copy()
        layer = cv2.circle(
            img=layer,
            center=self._safe_pt(pt1),
            color=color if color else self.color(None),
            thickness=thickness,
        )
        self.merge_visible(layer, alpha)

    def draw_box(self, pt1, pt2, alpha=1, thickness=1, color=None, title=None):
        # draw a bounding box / box with title on canvas
        layer = self.image.copy()
        layer = cv2.rectangle(
            img=layer,
            pt1=self._safe_pt(pt1),
            pt2=self._safe_pt(pt2),
            color=color if color else self.color(title),
            thickness=thickness,
            lineType=4,
        )
        # draw labels with auto-fitting background color
        if title:
            # draw background
            text_size, _ = cv2.getTextSize(title, self.font, self.font_scale, self.font_thickness)
            text_w, text_h = text_size
            x1, y1 = self._safe_pt(pt1)
            layer = cv2.rectangle(
                img=layer,
                pt1=(x1, y1),
                pt2=(x1 + text_w + 2, y1 + text_h + 2),
                color=color if color else self.color(title),
                thickness=-1,
            )
            # draw texts
            layer = cv2.putText(
                img=layer,
                text=title,
                org=(x1, y1 + text_h),
                fontFace=self.font,
                fontScale=self.font_scale,
                color=self.font_color,
                thickness=self.font_thickness,
            )
        self.merge_visible(layer, alpha)

    def save(self, filename):
        cv2.imwrite(filename, self.image)

    def show(self, title="test", wait_key=False):
        if self.backend == "cv2":
            cv2.imshow(title, self.image)
            cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
            if wait_key:
                cv2.waitKey(0)
        elif self.backend == "plt":
            plt.imshow(self.image[:, :, ::-1], aspect='auto')
            plt.axis("off")
            plt.show()
        else:
            raise NotImplementedError
