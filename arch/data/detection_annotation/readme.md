## Get Started
### quick preview
```python
from arch.data.detection_annotation import Yolo5Dataloader, coco_meta_data
from arch.ia import Canvas

if __name__ == "__main__":
    dataloader = Yolo5Dataloader("../datasets/coco128/images/train2017", image_size=640, batch_size=4, training=True)
    canvas = Canvas()
    for image_batch, labels_batch in dataloader:
        for i, image in enumerate(image_batch):
            labels = labels_batch[labels_batch[:, 0] == i, 1:]
            canvas.load(image).draw_boxes(labels, image.shape[-2:], class_names=coco_meta_data.names)
            canvas.show(wait_key=True)
```

### auto-anchor
```python
from arch.data.detection_annotation import Yolo5Dataloader

if __name__ == "__main__":
    dataloader = Yolo5Dataloader("../datasets/coco128/images/train2017", image_size=640, batch_size=4, training=True)
    anchor = dataloader.auto_anchor()
    print(anchor)
```