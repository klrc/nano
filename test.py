import torch
import nano
from nano.detection import CaffeWrapper, evaluator

# load from pytorch
model = nano.models.yolox_cspm_depthwise(num_classes=4)
model.load_state_dict(torch.load("runs/train/exp130/weights/last.pt", map_location="cpu")["state_dict"])
model.names = ["person", "bike", "car", "misc"]
imgsz = 416

# load from caffe
# root = "release/yolox_cspm_depthwise_test"
# model = CaffeWrapper(
#     caffemodel_path=f"{root}/yolox_cspm.caffemodel",
#     prototxt_path=f"{root}/yolox_cspm.prototxt",
#     output_names=["output.1", "output.2", "output.3"],
#     class_names=["person", "two-wheeler", "car"],
#     anchors=[[10.875, 14.921875], [31.1875, 53.28125], [143.0, 157.5]],
# )
# imgsz=[224, 416]

evaluator = nano.detection.evaluator
evaluator.run(
    model,
    data="configs/coc-misc-s.yaml",
    batch_size=1,
    imgsz=imgsz,
    device="cpu",
)
