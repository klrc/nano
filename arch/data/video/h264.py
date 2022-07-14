import cv2


class H264VideoStream:
    def __init__(self, file_path, dsize=None, color_format=None):
        self.file_path = file_path
        self.dsize = dsize
        self.color_format = color_format
        self.cap = None

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration
        if self.dsize:
            frame = cv2.resize(frame, self.dsize)
        if self.color_format:
            frame = cv2.cvtColor(frame, self.color_format)
        return frame

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_path)
        assert self.cap.isOpened()
        return self
