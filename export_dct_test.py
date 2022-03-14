import os
import shutil
import torch
from loguru import logger
from torch.onnx import OperatorExportTypes
from nano.models.multiplex.dct import DCTModule
from xnnc.docker import caffe_builder

if __name__ == "__main__":
    # model setup ======================================================================================
    logger.debug("Loading pytorch neural network model")
    # make sure to configure the following variables
    model_stamp = "dct_test"
    model = DCTModule(8)
    input_size = (1, 3, 256, 512)
    forced_export = True
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


    # Generate ONNX models =============================================================================
    logger.debug("Build ONNX model")
    onnx_path = f"{root}/{model_stamp}.onnx"
    torch.onnx.export(
        model,
        args=torch.rand(*input_size),
        f=onnx_path,
        input_names=["input"],
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

    logger.debug("All process finished")
