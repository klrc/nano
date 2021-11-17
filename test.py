import torch
import nano
from nano.detection import CaffeWrapper, evaluator

# load from pytorch
# model = nano.models.yolox_esmk_shrink_misc(num_classes=4)
# model.load_state_dict(torch.load("runs/train/exp130/weights/last.pt", map_location="cpu")["state_dict"])
# model.names = ["person", "bike", "car"]
# imgsz = 416

# load from caffe
root = "release/yolox_esmk_v2"
model = CaffeWrapper(
    caffemodel_path=f"{root}/yolox_esmk_shrink.caffemodel",
    prototxt_path=f"{root}/yolox_esmk_shrink.prototxt",
    output_names=["output_1", "output_2", "output_3"],
    class_names=["person", "bike", "car"],
    anchors=[[11.3359375, 13.6875], [30.359375, 46.34375], [143.0, 129.5]],
)
imgsz=[224, 416]

evaluator = nano.detection.evaluator
evaluator.run(
    model,
    data="configs/coco-val.yaml",
    batch_size=1,
    imgsz=imgsz,
    device="cpu",
)
