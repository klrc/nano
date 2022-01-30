from operator import index
import os
import random
from typing import Iterable, List

from matplotlib.pyplot import cla

import nano.data.transforms as T
from torch.utils.data import DataLoader, Dataset


def file_io_layer(root):
    # extract files from specified root
    if os.path.exists(root):
        for fname in os.listdir(root):
            yield fname, f"{root}/{fname}"


def detection_data_layer(*roots):
    # generate detction dataset items
    unpaired = {}
    for root in roots:
        for name, path in file_io_layer(root):
            name, _ = name.split(".")
            if name in unpaired:
                yield unpaired.pop(name), path
            else:
                unpaired[name] = path


class Assembly(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = []
        self.hooks = []

    def append_data(self, *data_generators: Iterable):
        for generator in data_generators:
            self.data += [x for x in generator]
        assert len(self.data) > 0, "No samples found in layer"

    def append(self, hook_fn):
        self.hooks.append(hook_fn)

    def compose(self, *hooks):
        for hook_fn in hooks:
            self.append(hook_fn)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.__chain__(index, self.hooks)

    def __chain__(self, index, hooks: List[T.TransformFunction]):
        if len(hooks) == 0:
            return self.data[index]
        hook_fn = hooks[-1]
        if random.random() <= hook_fn.p:
            if hook_fn.feed_samples > 1:
                index_list = [index]
                index_list += [random.randint(0, len(self) - 1) for _ in range(hook_fn.feed_samples - 1)]
                data = [self.__chain__(x, hooks[:-1]) for x in index_list]
            else:
                data = self.__chain__(index, hooks[:-1])
            return hook_fn(data)
        else:
            return self.__chain__(index, hooks[:-1])

    def as_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)
