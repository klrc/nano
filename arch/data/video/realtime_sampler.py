from copy import deepcopy
from multiprocessing import Process, Queue
import time
from typing import Iterable


class RealtimeSampler:
    def __init__(self, source: Iterable, fps=24, queue_size=2):
        self.source = source
        self.fps = fps
        self.pipe = Queue(maxsize=queue_size)

    @staticmethod
    def work_process(pipe: Queue, source, fps):
        source = deepcopy(source)
        for frame in source:
            if not pipe.full():
                pipe.put(frame)
            time.sleep(1 / fps)
        # Ending flag
        pipe.put(None)

    def __next__(self):
        frame = self.pipe.get()
        if frame is None:
            raise StopIteration
        return frame

    def __iter__(self):
        # Create a virtual video stream
        proc = Process(target=self.work_process, args=(self.pipe, self.source, self.fps))
        proc.daemon = True
        proc.start()
        return self
