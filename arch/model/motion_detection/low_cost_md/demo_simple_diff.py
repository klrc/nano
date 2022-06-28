import numpy as np
import cv2


class MDASimpleDiff:
    def __init__(self, size) -> None:
        self.frame_height, self.frame_width = size
        self.chunk_size = 4
        self.buffer_0 = np.zeros((self.frame_height // self.chunk_size, self.frame_width // self.chunk_size))
        self.buffer_1 = np.zeros((self.frame_height // self.chunk_size, self.frame_width // self.chunk_size))
        self.main_buffer = 0
        self.dilation = 4

    def __call__(self, frame):
        # RGB to GRAY
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Load attributes (will be simplified in DSP)
        chunk_size = self.chunk_size
        frame_height, frame_width = self.frame_height, self.frame_width
        dilation = self.dilation

        # Switch main memory buffer
        if self.main_buffer == 0:
            memory = self.buffer_0
            diff = self.buffer_1
        else:
            memory = self.buffer_1
            diff = self.buffer_0
        self.main_buffer = 1 - self.main_buffer

        # Clear MEMORY & CHUNK SUM
        scale_factor = chunk_size ** 2 / dilation ** 2
        for i in range(frame_height // chunk_size):
            for j in range(frame_width // chunk_size):
                memory[i, j] = 0
                for ri in range(chunk_size):
                    for rj in range(chunk_size):
                        if ri % dilation == 0 and rj % dilation == 0:
                            memory[i, j] += frame[i * chunk_size + ri, j * chunk_size + rj] / scale_factor

        # Calculate DIFF
        for i in range(frame_height // chunk_size):
            for j in range(frame_width // chunk_size):
                dif_value = np.abs(diff[i, j] - memory[i, j])
                if dif_value > 10:
                    dif_value = 255
                diff[i, j] = dif_value

        # Quantize
        return diff.astype(np.uint8)

