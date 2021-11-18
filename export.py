# items to export:
# ---------------------- raw
#  - xxx.py
#  - xxx.pt
# ---------------------- reports
#  - readme.yaml
#       - class_names
#       - output_names
#       - anchors
#       - test_log (optional)
# ---------------------- porting
#  - xxx.onnx
#  - xxx.caffemodel
#  - xxx.prototxt
# ----------------------- xnnc only
#  - xxx-custom.prototxt
#

import os
import shutil
import sys

os.environ["GLOG_minloglevel"] = "2"

import torch
import nano
from nano._utils import freeze


# replace Deconv layer with XNNC mResize layer, add final output layer
def switch_to_xnnc_layer(proto_path, output_names):
    with open(proto_path, "r") as f:
        layers = []
        current_layer = None
        for line in f.readlines():
            if line.startswith("layer {"):
                if current_layer is not None:
                    layers.append(current_layer)  # push last layer
                current_layer = []
            current_layer.append(line)
        layers.append(current_layer)
    for layer in layers:
        if "Deconvolution" in layer[2]:  # style
            layer[1] = layer[1].replace('name: "', 'name: "_')  # change name for param issue
            layer[2] = layer[2].replace("Deconvolution", "CppCustom")
            end_closure = layer[-1]
            while len(layer) > 5:
                layer.pop(-1)
            layer.append("  cpp_custom_param {\n")
            layer.append('    module: "mResize"\n')
            layer.append('    param_map_str: "scaleX:2 scaleY:2 align_corners:1"\n')
            layer.append("  }\n")
            layer.append(end_closure)
    layer = []
    layer.append("layer {\n")
    layer.append('  name: "detection_out"\n')
    layer.append('  type: "CppCustom"\n')
    for output_name in output_names:
        layer.append(f'  bottom: "{output_name}"\n')
    layer.append('  top: "detection_out"\n')
    layer.append("  cpp_custom_param {\n")
    layer.append('    module: "XnncMobiYoloOutputLayer"\n')
    layer.append(
        '    param_map_str: "num_classes:3 share_location:1 background_label_id:0 nms_threshold:0.45 top_k:400 keep_top_k:200 confidence_threshold:0.25"\n'
    )
    layer.append("  }\n")
    layer.append("}\n")
    layers.append(layer)
    with open(proto_path, "w") as f:
        for layer in layers:
            for line in layer:
                f.write(line)


def export(
    model,
    model_stamp,
    output_names,
    class_names,
    force=False,
):
    root = f"release/{model_stamp}"
    if os.path.exists(root):
        if force:
            print("rm", root)
            shutil.rmtree(root)
        else:
            raise FileExistsError("please check your release model stamp.")
    os.makedirs(root)
    python_source = os.path.abspath(sys.modules[model.__module__].__file__)
    target_source = f'{root}/{python_source.split("/")[-1]}'
    print("build", target_source)
    shutil.copy(python_source, target_source)  # copy source .py file

    model_name = "-".join(model_stamp.split("-")[:-1])
    target_pt = f"{root}/{model_name}.pt"
    print("build", target_pt)
    torch.save(model.state_dict(), target_pt)  # save .pt file

    target_readme = f"{root}/readme.yaml"
    print("build", target_readme)
    with open(target_readme, "w") as f:  # save readme.yaml configuration file
        values = dict(
            class_names=class_names,
            output_names=output_names,
            anchors=list(model.head.anchor_grid.flatten().numpy()),
        )
        for k, v in values.items():
            f.write(f"{k}: {v}\n")

    # save .onnx & .caffemodel & .prototxt file
    target_onnx = f"{root}/{model_name}.onnx"
    freeze(
        model,
        onnx_path=target_onnx,
        to_caffe=True,
        check_consistency=False,
        output_names=output_names,
    )

    # save *-custom.prototxt file
    target_custom = f"{root}/{model_name}-custom.prototxt"
    print("build", target_custom)
    shutil.copy(f"{root}/{model_name}.prototxt", target_custom)
    switch_to_xnnc_layer(target_custom, output_names)


if __name__ == "__main__":
    # model setup
    model = nano.models.yolox_esmk_shrink(num_classes=3).dsp()
    model.load_state_dict(torch.load("runs/train/exp150/weights/best.pt", map_location="cpu")["state_dict"])
    model_stamp = "yolox-esmk-2.18"
    output_names = ["output_1", "output_2", "output_3"]
    class_names = ["person", "bike", "car"]

    export(
        model,
        model_stamp=model_stamp,
        output_names=output_names,
        class_names=class_names,
        # force=True,
    )
