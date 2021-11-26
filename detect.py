import nano
import torch
import yaml
from nano.detection import CaffeWrapper

# model = nano.models.yolox_esmk_shrink(num_classes=3)
# model.load_state_dict(torch.load("release/yolox-esmk-2.19/yolox-esmk.pt", map_location="cpu"))
# model.names = ["person", "bike", "car"]
# imgsz = 416

# load from caffe
root = "release/yolox-esmk-2.25"
model_stamp = root.split("/")[-1]
model_name = "-".join(model_stamp.split("-")[:-1])
with open(f'{root}/readme.yaml', 'r') as f:
    for line in f.readlines():
        if not line.strip().startswith('#') and ':' in line:
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

detector = nano.detection.detector

detector.run(
    model,
    source="../datasets/tmpvoc",
    imgsz=imgsz,
    device="cpu",
)
