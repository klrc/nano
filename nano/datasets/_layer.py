import torch
from torch.utils.data import Dataset


class DatasetLayer(Dataset):
    def __init__(self, base=None, pre_layer_type=None) -> None:
        super().__init__()
        if pre_layer_type is not None:
            assert isinstance(base, pre_layer_type)
        self.base = base

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
