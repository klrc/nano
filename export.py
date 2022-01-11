import os
import shutil
import sys
import torch.nn as nn
import zipfile
import time
import torch
from nano.models.backbones.enhanced_shufflenet_v2 import ESBlockS1
from nano.models.heads.nanodet_head import NanoHeadless
import xnnc


if __name__ == "__main__":
    # model setup ===========================================
    print("\n[model setup]")
    from nano.models.model_zoo.nano_ghost import GhostNano_3x3_s32

    model = GhostNano_3x3_s32(num_classes=3)
    model.load_state_dict(torch.load("runs/train/exp119/last.pt", map_location="cpu")["state_dict"])
    model.head = NanoHeadless(model.head)
    print("adjust model..")
    for m in model.modules():
        if isinstance(m, ESBlockS1):
            m.channel_shuffle = nn.ChannelShuffle(2)
    model.eval()

    output_names = ["output_0", "output_1", "output_2"]
    class_names = ["person", "bike", "car"]
    model_stamp = "GhostNano_3x3_s32"
    forced_export = True
    print("start exporting", model_stamp, "..")

    # Copy source file for backup ===========================
    print("build release dir ..")
    root = f"release/{model_stamp}"
    if os.path.exists(root):
        if forced_export:
            # print("rm", root)
            shutil.rmtree(root)
        else:
            raise FileExistsError("please check your release model stamp.")
    os.makedirs(root)

    # Save .pt file =========================================
    target_pt = f"{root}/{model_stamp}.pt"
    torch.save(model.state_dict(), target_pt)  # save .pt file
    print("saving pt file as", target_pt)

    # Save .yaml configuration ==============================
    print("\n[build release]")
    target_readme = f"{root}/readme.txt"
    print("generate", target_readme)
    with open(target_readme, "w") as f:  # save readme.txt configuration file
        f.write(f"build_time: {time.asctime(time.localtime(time.time()))}\n")
        f.write(f"output_names: {output_names}\n")
        f.write(f"class_names: {class_names}\n")

    # # Generate onnx & caffe models ==========================
    print("\n[build onnx & caffe model]")
    onnx_path, caffemodel_path, prototxt_path = xnnc.make(
        model,
        model_stamp,
        cache_dir=root,
        output_names=output_names,
        dummy_input_shape=(1, 3, 224, 416),
    )

    # # Add final PP layer (optional) =========================
    # print("adjust final prototxt")
    # with open(prototxt_path, "a") as f:
    #     f.write("layer {\n")
    #     f.write('  name: "detection_out"\n')
    #     f.write('  type: "CppCustom"\n')
    #     for output_name in output_names:
    #         f.write(f'  bottom: "{output_name}"\n')
    #     f.write('  top: "detection_out"\n')
    #     f.write("  cpp_custom_param {\n")
    #     f.write('    module: "yoloxpp"\n')
    #     f.write('    param_map_str: ""\n')
    #     f.write("  }\n")
    #     f.write("}\n")

    # # Directly porting to XNNC source (optional) =====================================
    # print("\n[update XNNC source]")
    # for cmd in [
    #     "rm xnnc/src/Example/yolox-series/model/*.prototxt",
    #     "rm xnnc/src/Example/yolox-series/model/*.caffemodel",
    #     f"cp release/{model_stamp}/*-custom.prototxt xnnc/src/Example/yolox-series/model/",
    #     f"cp release/{model_stamp}/*.caffemodel xnnc/src/Example/yolox-series/model/",
    # ]:
    #     print(cmd)
    #     os.system(cmd)

    # print("\nall process finished.")
