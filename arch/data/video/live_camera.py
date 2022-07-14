import cv2


def uniform_scale(image, inf_size):
    height, width, _ = image.shape
    ratio = inf_size / max(height, width)
    image = cv2.resize(image, (int(ratio * width), int(ratio * height)))
    return image


class LiveCamera:
    def __init__(self, inf_size=None, exit_key="q", horizontal_flip=True):
        self.cap = None
        self.inf_size = inf_size
        self.exit_key = exit_key
        self.horizontal_flip = horizontal_flip

    def __iter__(self):
        self.cap = cv2.VideoCapture(0)
        assert self.cap.isOpened()
        # size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # print("size: " + repr(size))
        # warmup
        _ = self.cap.read()
        _ = self.cap.read()
        _ = self.cap.read()
        return self

    def __next__(self):
        # Press ESC to break
        key = cv2.waitKey(1) & 0xFF
        if key == ord(self.exit_key):
            self.cap.release()
            raise StopIteration
        frame = None
        while frame is None:
            success, frame = self.cap.read()
            if not success:
                self.cap.release()
                raise Exception("camera connection lost")
        if self.inf_size:
            frame = uniform_scale(frame, self.inf_size)
        if self.horizontal_flip:
            frame = cv2.flip(frame, 1)
        return frame
