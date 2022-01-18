import os
import shutil
import torch.nn as nn
import time
import torch
from nano.models.backbones.enhanced_shufflenet_v2 import ESBlockS1
from nano.models.heads.nanodet_head import NanoHeadless
from nano.models.model_zoo.nano_ghost import GhostNano_3x4_m96
import xnnc


if __name__ == "__main__":
    # model setup ===========================================
    print("INFO: Loading pytorch neural network model")
    # make sure to configure the following variables
    model_stamp = "GhostNano_3x4_m96_build_test"
    output_names = ["output_0", "output_1", "output_2", "output_3"]
    class_names = ["person", "bike", "car"]
    num_strides = len(output_names)
    model = GhostNano_3x4_m96(num_classes=len(class_names))
    model.load_state_dict(torch.load("release/GhostNano_3x4_m96/GhostNano_3x4_m96.pt", map_location="cpu"))
    input_size = (224, 416)
    forced_export = True
    # -------------------------------------------
    print("INFO: Replacing layer for support: ChannelShuffle")
    model.head = NanoHeadless(model.head)
    for m in model.modules():
        if isinstance(m, ESBlockS1):
            m.channel_shuffle = nn.ChannelShuffle(2)
    model.eval()

    # Copy source file for backup ===========================
    print("INFO: Exporting model as", model_stamp, "..")
    root = f"release/{model_stamp}"
    print("INFO: Build release dir:", root)
    if os.path.exists(root):
        if forced_export:
            print("INFO: Forced exporting, removing", root)
            shutil.rmtree(root)
        else:
            raise FileExistsError("please check your release model stamp.")
    os.makedirs(root)

    # Save .pt file =========================================
    target_pt = f"{root}/{model_stamp}.pt"
    torch.save(model.state_dict(), target_pt)  # save .pt file
    print("INFO: Saving pt file as", target_pt)

    # Save .txt configuration ==============================
    target_readme = f"{root}/readme.txt"
    print("INFO: Generate release information:", target_readme)
    with open(target_readme, "w") as f:  # save readme.txt configuration file
        f.write(f"build_time: {time.asctime(time.localtime(time.time()))}\n")
        f.write(f"output_names: {output_names}\n")
        f.write(f"class_names: {class_names}\n")

    # Generate onnx & caffe models ==========================
    print("INFO: Build onnx & caffe model")
    onnx_path, caffemodel_path, prototxt_path = xnnc.build(
        model,
        model_stamp,
        cache_dir=root,
        output_names=output_names,
        dummy_input_shape=(1, 3, input_size[0], input_size[1]),
    )
    # Add final PP layer (optional)
    with open(prototxt_path, "a") as f:
        f.write("layer {\n")
        f.write('  name: "detection_out"\n')
        f.write('  type: "CppCustom"\n')
        for output_name in output_names:
            f.write(f'  bottom: "{output_name}"\n')
        f.write('  top: "detection_out"\n')
        f.write("  cpp_custom_param {\n")
        f.write('    module: "yoloxpp"\n')
        f.write('    param_map_str: ""\n')
        f.write("  }\n")
        f.write("}\n")

    # Directly porting to XNNC source =====================================
    print("INFO: Updating XNNC source ..")
    xnnc_root = "/home/sh/Projects/tensilica/xtensa/XNNC"
    xnnc_project = "yolox-series"
    for cmd in [
        f"sudo rm -r {xnnc_root}/Example/{xnnc_project}",
        f"mkdir {xnnc_root}/Example/{xnnc_project}",
        f"mkdir {xnnc_root}/Example/{xnnc_project}/model",
        f"mkdir {xnnc_root}/Example/{xnnc_project}/layers",
        f"cp release/{model_stamp}/*-custom.prototxt {xnnc_root}/Example/{xnnc_project}/model/yolox.prototxt",
        f"cp release/{model_stamp}/*.caffemodel {xnnc_root}/Example/{xnnc_project}/model/yolox.caffemodel",
        f"cp -r xnnc/cpp_layers/* {xnnc_root}/Example/{xnnc_project}/layers/",
        f"cp xnnc/yolox.cfg {xnnc_root}/Example/{xnnc_project}/",
    ]:
        print(cmd)
        os.system(cmd)
    print("INFO: Generating meanval.txt")
    with open(f"{xnnc_root}/Example/{xnnc_project}/model/meanval.txt", "w") as f:
        f.write("3    # channels\n")
        f.write(f"{input_size[0]}    # height\n")
        f.write(f"{input_size[1]}    # width\n")
        f.write("0.0    # channel 1\n")
        f.write("0.0    # channel 2\n")
        f.write("0.0    # channel 3\n")
    print("INFO: Generating output_ctrl.txt")
    with open(f"{xnnc_root}/Example/{xnnc_project}/model/output_ctrl.txt", "w") as f:
        f.write("[detection_out:detection_out] save det dataset/voc2012_labels.txt dataset/")
    print("INFO: all process finished")

    # Build test -----------------------
    from test_xnnc_build import yolox_test
    yolox_test()