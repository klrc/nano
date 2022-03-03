import numpy as np
import torch
import cv2

from nano.data.transforms import Tensor2Image, TransformFunction


def any_image_format(image):
    if isinstance(image, torch.Tensor):
        return Tensor2Image.functional(image)
    return image


def any_tensor_format(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    return tensor


def random_color(default=None):
    if default is None:
        return list(np.random.random(size=3) * 128 + 128)  # light color
    return default


class Canvas:
    def __init__(self, any_image) -> None:
        self.image = any_image_format(any_image)
        self._color = self.next_color()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.3
        self.font_thickness = 1
        self.font_color = (0, 0, 0)

    def set_color(self, rgb):
        R, G, B = rgb  # RGB -> cv2 BGR
        self._color = (B, G, R)

    def next_color(self):
        self._color = list(np.random.random(size=3) * 128 + 128)  # light color
        return self._color

    def merge_transparent_layer(self, layer, alpha):
        alpha = float(alpha)
        if alpha < 1:
            return cv2.addWeighted(self.image, 1 - alpha, layer, alpha, 0)
        return layer

    def draw_point(self, center, thickness=3, alpha=1):
        # draw square pixel-style points on image
        layer = self.image.copy()
        center = [int(x) for x in center]
        layer = cv2.circle(layer, center, 1, self._color, thickness)
        self.image = self.merge_transparent_layer(layer, alpha)

    def draw_boxes(self, boxes, alpha=1, thickness=1, color=None):
        if color is not None:
            self.set_color(color)
        for box in boxes:
            self.draw_box(box, alpha, thickness)

    def draw_box(self, box, alpha=1, thickness=1):
        layer = self.image.copy()
        x1, y1, x2, y2 = [int(x) for x in box]
        layer = cv2.rectangle(layer, (x1, y1), (x2, y2), self._color, thickness, 4, 0)
        self.image = self.merge_transparent_layer(layer, alpha)

    def draw_text_with_background(self, text, top_left_point, alpha=1):
        # draw labels with auto-fitting background color
        text_size, _ = cv2.getTextSize(text, self.font, self.font_scale, self.font_thickness)
        text_w, text_h = text_size
        layer = self.image.copy()
        x1, y1 = [int(x) for x in top_left_point]
        layer = cv2.rectangle(layer, (x1, y1), (x1 + text_w + 2, y1 + text_h + 2), self._color, -1)
        layer = cv2.putText(layer, text, (x1, y1 + text_h), self.font, self.font_scale, self.font_color, self.font_thickness)
        self.image = self.merge_transparent_layer(layer, alpha)


# modulized as <TransformFunction> for quick testing implementation
class RenderLabels(TransformFunction):
    def __init__(self, class_names) -> None:
        super().__init__(p=1)
        self.class_names = class_names

    def __call__(self, data):
        """
        image: image with cv2(numpy) format or pytorch tensor format
        label: labels nx5 (cid, xyxy)
        """
        image, label = data
        return self.functional(image, label, self.class_names)

    @staticmethod
    def functional(image, label, class_names):
        """
        image: image with cv2(numpy) format or pytorch tensor format
        label: labels nx5 (cid, xyxy)
        """
        canvas = Canvas(image)
        if label.size(-1) == 5:  # nx5
            for obj_arr in label:
                c, x1, y1, x2, y2 = [int(x) for x in obj_arr]
                text = class_names[c]
                canvas.draw_box((x1, y1, x2, y2))
                canvas.draw_text_with_background(text, (x1, y1))
                canvas.next_color()
        else:
            raise NotImplementedError
        return canvas.image
