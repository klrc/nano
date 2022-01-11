import torch

def check_size(model):
    params = 0
    for module in model.modules():
        if hasattr(module, "weight") and hasattr(module.weight, "size"):
            params += torch.prod(torch.LongTensor(list(module.weight.size())))
        if hasattr(module, "bias") and hasattr(module.bias, "size"):
            params += torch.prod(torch.LongTensor(list(module.bias.size())))
    total_params_size = abs(params.numpy() * 4. / (1024 ** 2.))
    print("Params size (MB): %0.2f" % total_params_size)
