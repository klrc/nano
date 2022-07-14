## Get Started

### quick preview (frame-by-frame mode)
```python
from arch.data.video import H264VideoStream
import cv2

if __name__ == '__main__':
    video = H264VideoStream('../datasets/v1.5.6.5人形测试/12M左右不报.avi')
    for frame in video:
        cv2.imshow('test', frame)
        cv2.waitKey(0)
```

### quick preview (realtime mode)
```python
from arch.data.video import H264VideoStream, RealtimeSampler
import cv2

if __name__ == '__main__':
    video = H264VideoStream('../datasets/v1.5.6.5人形测试/12M左右不报.avi')
    sampler = RealtimeSampler(video, fps=24)
    for frame in sampler:
        cv2.imshow('test', frame)
        cv2.waitKey(1)
```

### quick preview (live camera)

```python
from arch.data.video import LiveCamera
import cv2

if __name__ == '__main__':
    video = LiveCamera(inf_size=480, exit_key='q')
    for frame in video:
        cv2.imshow('test', frame)
```