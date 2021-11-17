import torch
import nano
import yaml
from nano.detection import CaffeWrapper, evaluator

# load from pytorch
# model = nano.models.yolox_esmk_shrink_misc(num_classes=4)
# model.load_state_dict(torch.load("runs/train/exp130/weights/last.pt", map_location="cpu")["state_dict"])
# model.names = ["person", "bike", "car"]
# imgsz = 416

# load from caffe
root = "release/yolox-esmk-2.17"
model_stamp = root.split("/")[-1]
model_name = "-".join(model_stamp.split("-")[:-1])
with open(f'{root}/readme.yaml', 'r') as f:
    for line in f.readlines():
        for k, v in yaml.load(line).items():
            if k == 'output_names':
                output_names = v
            elif k == 'class_names':
                class_names = v
            elif k == 'anchors':
                anchors = v
                anchors = torch.tensor(anchors).reshape(-1, 2).numpy()

model = CaffeWrapper(
    caffemodel_path=f"{root}/{model_name}.caffemodel",
    prototxt_path=f"{root}/{model_name}.prototxt",
    output_names=output_names,
    class_names=class_names,
    anchors=anchors,
)
imgsz = [224, 416]

evaluator = nano.detection.evaluator
evaluator.run(
    model,
    data="configs/coco-val.yaml",
    batch_size=1,
    imgsz=imgsz,
    device="cpu",
)
