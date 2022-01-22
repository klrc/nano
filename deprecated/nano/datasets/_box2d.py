import torch
from torch.utils.data import Dataset


class DatasetLayer(Dataset):
    def __init__(self, base=None, *pre_layer_type) -> None:
        super().__init__()
        assertion = [isinstance(base, p) for p in pre_layer_type]
        assert len(pre_layer_type) == 0 or any(assertion), assertion
        self.base = base

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
