import os
import shutil
import torch.nn as nn
import time
import torch
from loguru import logger
from torch.onnx import OperatorExportTypes
from nano.models.backbones.enhanced_shufflenet_v2 import ESBlockS1
from nano.models.heads.nanodet_head import NanoHeadless
from nano.models.model_zoo.nano_ghost import GhostNano_3x4_m96
from xnnc.docker import caffe_builder, xnnc_builder
from nano.data.dataset_info import drive3_names


if __name__ == "__main__":
    # model setup ======================================================================================
    logger.debug("Loading pytorch neural network model")
    # make sure to configure the following variables
    model_stamp = "GhostNano_3x4_m96"
    output_names = ["output_0", "output_1", "output_2", "output_3"]
    class_names = drive3_names
    num_strides = len(output_names)
    model = GhostNano_3x4_m96(num_classes=len(class_names))
    model.load_state_dict(torch.load("runs/train/exp17/best.pt", map_location="cpu")["state_dict"])
    # model.load_state_dict(torch.load("release/GhostNano_3x4_m96/GhostNano_3x4_m96.pt", map_location="cpu"))
    input_size = (1, 3, 224, 416)
    forced_export = True
    # -------------------------------------------
    logger.debug("Replacing layer for support: ChannelShuffle")
    model.head = NanoHeadless(model.head)
    for m in model.modules():
        for cname, c in m.named_children():
            if isinstance(c, ESBlockS1):
                # setattr(m, cname, nn.Identity())
                c.channel_shuffle = nn.ChannelShuffle(2)
    model.eval()
    logger.success(f"Exporting model as {model_stamp}")

    # Copy source file for backup ======================================================================
    root = f"release/{model_stamp}"
    logger.debug(f"Build release dir: {root}")
    if os.path.exists(root):
        if forced_export:
            logger.debug(f"Forced exporting, removing {root}")
            shutil.rmtree(root)
        else:
            raise FileExistsError("please check your release model stamp.")
    os.makedirs(root)

    # Save .pt file ====================================================================================
    target_pt = f"{root}/{model_stamp}.pt"
    torch.save(model.state_dict(), target_pt)  # save .pt file
    logger.success(f"Saving pt file as {target_pt}")

    # Save .txt configuration =========================================================================
    target_readme = f"{root}/readme.txt"
    with open(target_readme, "w") as f:  # save readme.txt configuration file
        f.write(f"build_time: {time.asctime(time.localtime(time.time()))}\n")
        f.write(f"output_names: {output_names}\n")
        f.write(f"class_names: {class_names}\n")
    logger.success(f"Saving description file as {target_readme}")

    # Generate ONNX models =============================================================================
    logger.debug("Build ONNX model")
    onnx_path = f"{root}/{model_stamp}.onnx"
    torch.onnx.export(
        model,
        args=torch.rand(*input_size),
        f=onnx_path,
        input_names=["input"],
        output_names=output_names,
        opset_version=12,
        operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
        enable_onnx_checker=True,
        do_constant_folding=False,
        keep_initializers_as_inputs=True,
        verbose=True,
    )
    logger.success(f"Save ONNX model as {onnx_path}")

    # Generate Caffe models =============================================================================
    logger.debug("Build Caffe model & prototxt")
    project_root = "/Users/sh/Projects/nano"
    with caffe_builder(
        working_dir="/script",
        build_path=f"{project_root}/{root}",
        script_path=f"{project_root}/xnnc/pycaffe_layers/",
        image="caffe-python38-cpu:arm64",
    ) as s:
        s.exec_run(f"python3 build.py --onnx_file {project_root}/{onnx_path}", stream=True)

    assert os.path.exists(f"{root}/{model_stamp}.caffemodel")
    assert os.path.exists(f"{root}/{model_stamp}-custom.prototxt")
    logger.success(f"Saved {root}/{model_stamp}.caffemodel")

    # Add final PP layer
    with open(f"{root}/{model_stamp}-custom.prototxt", "a") as f:
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
    logger.success(f"Saved {root}/{model_stamp}-custom.prototxt")

    # Update XNNC source =====================================================================================
    logger.debug("Updating XNNC source ..")
    xnnc_root = "/Users/sh/Projects/tensilica/xtensa/XNNC"
    xnnc_project = "yolox-series"
    xnnc_valset = 'calib_dataset_416x'
    for cmd in [
        f"sudo rm -r {xnnc_root}/Example/{xnnc_project}",
        f"mkdir {xnnc_root}/Example/{xnnc_project}",
        f"mkdir {xnnc_root}/Example/{xnnc_project}/model",
        f"mkdir {xnnc_root}/Example/{xnnc_project}/layers",
        f"cp {root}/*-custom.prototxt {xnnc_root}/Example/{xnnc_project}/model/yolox.prototxt",
        f"cp {root}/*.caffemodel {xnnc_root}/Example/{xnnc_project}/model/yolox.caffemodel",
        f"cp -r ../datasets/{xnnc_valset} {xnnc_root}/Example/{xnnc_project}/{xnnc_valset}",
        f"cp -r xnnc/cpp_layers/* {xnnc_root}/Example/{xnnc_project}/layers/",
        f"cp xnnc/yolox.cfg {xnnc_root}/Example/{xnnc_project}/",
    ]:
        logger.debug(cmd)
        os.system(cmd)
    logger.debug("Generating meanval.txt")
    with open(f"{xnnc_root}/Example/{xnnc_project}/model/meanval.txt", "w") as f:
        f.write("3    # channels\n")
        f.write(f"{input_size[-2]}    # height\n")
        f.write(f"{input_size[-1]}    # width\n")
        f.write("0.0    # channel 1\n")
        f.write("0.0    # channel 2\n")
        f.write("0.0    # channel 3\n")
    logger.debug("Generating output_ctrl.txt")
    with open(f"{xnnc_root}/Example/{xnnc_project}/model/output_ctrl.txt", "w") as f:
        f.write(f"[detection_out:detection_out] save det {xnnc_valset}/voc2012_labels.txt {xnnc_valset}/")
    logger.success(f"Update finished in {xnnc_root}")

    # Build DSP project =======================================================================================
    logger.debug("Build XNNC project ..")
    with xnnc_builder(
        working_dir="/xnnc/Example/yolox-series",
        xnnc_path="/Users/sh/Projects/tensilica/xtensa/XNNC",
        xtdev_path="/Users/sh/Projects/tensilica/xtensa/XtDevTools",
    ) as s:
        for custom_layer in ["yoloxpp", "slice"]:
            s.exec_run(f"cmake layers/{custom_layer}/CMakeLists.txt", stream=True)
            s.exec_run(f"make -C layers/{custom_layer}", stream=True)
            s.exec_run(f"cp layers/{custom_layer}/lib{custom_layer}.so ./", stream=True)
            s.exec_run(f"make install -C layers/{custom_layer}", stream=True)
        s.exec_run("python3 ../../Scripts/xnnc.py --keep --config_file yolox.cfg", stream=True)

    logger.debug("All process finished")
