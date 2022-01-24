from multiprocessing import get_context, Process
from multiprocessing.queues import Queue
import time
import random
import os
import sys


def send_to_hell(layer, maxsize=2):
    from multiprocessing.queues import Queue
    from multiprocessing import get_context

    queue = Queue(maxsize=maxsize, ctx=get_context())
    n = os.fork()
    if n <= 0:
        for data in layer:
            queue.put(data)
        queue.put(-1)
        sys.exit(0)

    def fetch_from_hell(queue):
        while True:
            data = queue.get()
            if data == -1:
                break
            yield data

    return fetch_from_hell(queue)


def stupid_generator():
    for i in range(10):
        yield f"data{i}"
        time.sleep(random.random())


layer = stupid_generator()
daemon = send_to_hell(layer, maxsize=2)

daemon = layer

import datetime

for x in daemon:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[0:23]  # 使用切片的方式使精度只到小数点后三位
    time.sleep(random.random()*2)
    print(now, x + " processed")
