import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor


def im2tensor(image: np.ndarray):
    assert np.issubsctype(image, np.integer)  # 0~255 BGR int -> 0~1 RGB float
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return to_tensor(image)


def tensor2im(x: torch.Tensor):
    assert isinstance(x, torch.Tensor) and len(x.shape) == 3  # 0~1 RGB float -> 0~255 BGR int
    np_img = (x * 255).int().numpy().astype(np.uint8)
    np_img = np_img[::-1].transpose((1, 2, 0))  # CHW to HWC, RGB to BGR, 0~1 to 0~255
    np_img = np.ascontiguousarray(np_img)
    return np_img


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
        return self

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
        return self

    def draw_point(self, pt1, thickness=3, radius=1, alpha=1, color=None):
        # draw a single point on canvas
        layer = self.image.copy()
        layer = cv2.circle(
            img=layer,
            center=self._safe_pt(pt1),
            radius=radius,
            color=color if color else self.color(None),
            thickness=thickness,
        )
        self.merge_visible(layer, alpha)
        return self

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
        return self

    def draw_boxes(self, nx5_cxywhn, image_size, alpha=1, thickness=1, color=None, class_names=None):
        height, width = image_size
        for c, x, y, w, h in nx5_cxywhn:
            x1, x2, y1, y2 = (x - w / 2) * width, (x + w / 2) * width, (y - h / 2) * height, (y + h / 2) * height
            x1, x2, y1, y2 = [int(x) for x in (x1, x2, y1, y2)]
            if class_names:
                title = class_names[int(c)]
            else:
                title = None
            self.draw_box((x1, y1), (x2, y2), alpha, thickness, color, title)
        return self

    def draw_heatmap(self, feature, alpha=0.5):
        h, w, _ = self.image.shape
        assert len(feature.shape) == 2
        feature = np.array(feature.abs())
        heatmap = None
        heatmap = cv2.normalize(feature, heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (w, h))
        self.merge_visible(heatmap, alpha)
        return self

    def save(self, filename):
        cv2.imwrite(filename, self.image)
        return self

    def show(self, title="test", wait_key=False):
        if self.backend == "cv2":
            cv2.imshow(title, self.image)
            cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
            if wait_key:
                cv2.waitKey(0)
        elif self.backend == "plt":
            plt.imshow(self.image[:, :, ::-1], aspect="auto")
            plt.axis("off")
            plt.show()
        else:
            raise NotImplementedError
        return self