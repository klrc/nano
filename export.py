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

import torch  # noqa: E402
import nano  # noqa: E402
from nano._utils import freeze  # noqa: E402


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
        if "_repflag_relu6" in layer[1]:
            layer[1] = layer[1].replace("_repflag_relu6", "")
            end_closure = layer.pop(-1)
            layer.append("  relu_param {\n")
            layer.append("    relu6_enable: 1\n")
            layer.append("  }\n")
            layer.append(end_closure)
        elif "Deconvolution" in layer[2]:  # style
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
    layer.append('    module: "yoloxpp"\n')
    layer.append('    param_map_str: ""\n')
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
    model = nano.models.mobilenet_v2_cspp_yolov5()
    model.load_state_dict(torch.load("runs/train/exp177/weights/best.pt", map_location="cpu")["state_dict"])
    model.head.mode_dsp_off = False
    model_stamp = "yolov5-mv2-1.2"
    output_names = ["fs0", "fs1", "fs2", "fs3"]
    class_names = ["person", "bike", "car"]

    export(
        model,
        model_stamp=model_stamp,
        output_names=output_names,
        class_names=class_names,
        force=True,
    )

    # optional
    os.system("rm nano/_utils/xnnc/Example/yolox-series/model/*.prototxt")
    os.system("rm nano/_utils/xnnc/Example/yolox-series/model/*.caffemodel")
    os.system(f"cp release/{model_stamp}/*-custom.prototxt nano/_utils/xnnc/Example/yolox-series/model/")
    os.system(f"cp release/{model_stamp}/*.caffemodel nano/_utils/xnnc/Example/yolox-series/model/")
