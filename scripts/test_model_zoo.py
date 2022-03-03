import torch
from loguru import logger
import sys
from thop import profile
from copy import deepcopy

sys.path.append(".")


def get_model_info(model, tsize):
    img = torch.rand(tsize, device=next(model.parameters()).device)
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    logger.success(f"Estimated Size:{params:.2f}M, Estimated Bandwidth: {flops * 2:.2f}G, Resolution: {tsize[2:]}")


if __name__ == "__main__":
    logger.debug("Model Zoo Test ------------------")

    from nano.models.model_zoo import (
        GhostNano_3x3_s64,
        GhostNano_3x4_s64,
        GhostNano_3x3_m96,
        GhostNano_3x4_m96,
        GhostNano_3x3_l128,
        GhostNano_3x4_l128,
    )

    get_model_info(GhostNano_3x3_s64(4).eval(), (1, 3, 416, 736))
    get_model_info(GhostNano_3x4_s64(4).eval(), (1, 3, 416, 736))
    get_model_info(GhostNano_3x3_m96(4).eval(), (1, 3, 288, 512))
    get_model_info(GhostNano_3x4_m96(4).eval(), (1, 3, 288, 512))
    get_model_info(GhostNano_3x3_l128(4).eval(), (1, 3, 224, 416))
    get_model_info(GhostNano_3x4_l128(4).eval(), (1, 3, 224, 416))
