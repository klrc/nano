import torch
from loguru import logger
from nano.models.model_zoo import (
    GhostNano_3x3_s64,
    GhostNano_3x4_s64,
    GhostNano_3x3_m96,
    GhostNano_3x4_m96,
    GhostNano_3x3_l128,
    GhostNano_3x4_l128,
)
from thop import profile
from copy import deepcopy


def test_model_zoo():
    logger.debug("Model Zoo Test ------------------")

    def get_model_info(model, tsize):
        img = torch.zeros(tsize, device=next(model.parameters()).device)
        flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
        params /= 1e6
        flops /= 1e9
        logger.success(f"Estimated Size:{params:.2f}M, Estimated Bandwidth: {flops * 2:.2f}G, Resolution: {tsize[2:]}")

    get_model_info(GhostNano_3x3_s64(26).eval(), (1, 3, 416, 736))
    get_model_info(GhostNano_3x4_s64(26).eval(), (1, 3, 416, 736))
    get_model_info(GhostNano_3x3_m96(26).eval(), (1, 3, 288, 512))
    get_model_info(GhostNano_3x4_m96(26).eval(), (1, 3, 288, 512))
    get_model_info(GhostNano_3x3_l128(26).eval(), (1, 3, 224, 416))
    get_model_info(GhostNano_3x4_l128(26).eval(), (1, 3, 224, 416))
