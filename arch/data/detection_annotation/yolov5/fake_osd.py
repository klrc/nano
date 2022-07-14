import random
import time

import cv2


class FakeOSD():
    def __init__(self):
        pass

    def __call__(self, image, labels):
        # random timestamp
        start = 189273600  # 1976-01-01 00：00：00
        end = 4133951999  # 1990-12-31 23：59：59
        t = random.randint(start, end)
        date_touple = time.localtime(t)
        possible_formats = {
            "ymd": ("%Y-%m-%d ",),
            "week": ("%a ", "%A "),
            "time": ("%H:%M:%S ",),
            "color": ((0, 0, 0), (255, 255, 255), (255, 255, 255), (255, 255, 255)),  # 75% white
            "font": (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_COMPLEX),
        }
        format = ""
        format += random.choice(possible_formats["ymd"])
        format += random.choice(possible_formats["week"])
        format += random.choice(possible_formats["time"])
        text = time.strftime(format, date_touple)
        text_color = random.choice(possible_formats["color"])
        x1, y1 = (
            random.randint(4, 64),  # x1
            random.randint(4, 32),  # y1
        )
        font = random.choice(possible_formats["font"])
        font_scale = random.random() * 0.4 + 0.4  # 0.4~0.8
        text_size, _ = cv2.getTextSize(text, font, font_scale, 1)
        _, text_h = text_size
        return cv2.putText(image, text, (x1, y1 + text_h), font, font_scale, text_color), labels
