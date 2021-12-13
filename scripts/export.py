import os
import shutil
import sys
import torch.nn as nn

import torch
import nano
import xnnc


if __name__ == "__main__":
    # model setup ===========================================
    print("\n[model setup]")
    model = nano.models.esnet_cspp_yolov5()
    model.load_state_dict(torch.load("runs/train/exp196/weights/best.pt", map_location="cpu")["state_dict"])

    output_names = ["fs1", "fs2", "fs3"]
    class_names = ["person", "bike", "car"]
    model_stamp = "yolov5-es-2.13"
    forced_export = False
    print("start exporting", model_stamp)

    # Process the model in advance ==========================
    print("adjust model")
    for m in model.modules():
        if isinstance(m, nano.models.yolomk.ESBlockS1):
            m.channel_shuffle = nn.ChannelShuffle(2)
    model.head.mode_dsp_off = False
    model.eval()

    # Copy source file for backup ===========================
    print("\n[build backup sources]")
    root = f"release/{model_stamp}"
    if os.path.exists(root):
        if forced_export:
            # print("rm", root)
            shutil.rmtree(root)
        else:
            raise FileExistsError("please check your release model stamp.")
    os.makedirs(root)
    python_source = os.path.abspath(sys.modules[model.__module__].__file__)
    target_source = f'{root}/{python_source.split("/")[-1]}'
    shutil.copy(python_source, target_source)  # copy source .py file

    # Save .pt file =========================================
    model_name = "-".join(model_stamp.split("-")[:-1])
    target_pt = f"{root}/{model_name}.pt"
    torch.save(model.state_dict(), target_pt)  # save .pt file
    print("saving pt file as", target_pt)

    # Save .yaml configuration ==============================
    print("\n[build YAML configuration]")
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

    # Generate onnx & caffe models ==========================
    print("\n[build onnx & caffe model]")
    onnx_path, caffemodel_path, prototxt_path = xnnc.make(
        model,
        model_name,
        cache_dir=root,
        output_names=output_names,
        dummy_input_shape=(1, 3, 224, 416),
    )

    # Add final PP layer (optional) =========================
    print("adjust final prototxt")
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
