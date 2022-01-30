import torch
from loguru import logger
from nano.models.model_zoo import *
from thop import profile
from copy import deepcopy

if __name__ == "__main__":

    logger.debug('Model Zoo Test ------------------')

    def get_model_info(model, tsize):
        # stride = 32
        img = torch.zeros(tsize, device=next(model.parameters()).device)
        flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
        params /= 1e6
        flops /= 1e9
        logger.success("Params: {:.2f}M, Gflops: {:.2f}".format(params, flops))
        logger.success("Estimated Size:{:.2f}M, Estimated Bandwidth: {:.2f}G".format(params, flops*2))

    get_model_info(GhostNano_3x3_s64(26).eval(), (1, 3, 416, 736))
    get_model_info(GhostNano_3x4_s64(26).eval(), (1, 3, 416, 736))
    get_model_info(GhostNano_3x3_m96(26).eval(), (1, 3, 288, 512))
    get_model_info(GhostNano_3x4_m96(26).eval(), (1, 3, 288, 512))
    get_model_info(GhostNano_4x3_m96(26).eval(), (1, 3, 224, 416))
    get_model_info(GhostNano_3x3_l128(26).eval(), (1, 3, 224, 416))
