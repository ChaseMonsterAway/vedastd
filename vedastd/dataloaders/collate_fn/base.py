from collections import OrderedDict

import numpy as np
import torch

from .registry import COLLATE_FN


@COLLATE_FN.register_module
class BaseCollate:
    def __init__(self, stack_keys):
        self.stack_keys = stack_keys

    def __call__(self, batch):
        data_dict = OrderedDict()
        for sample in batch:
            for k, v in sample.items():
                if k not in data_dict:
                    data_dict[k] = []
                if isinstance(v, np.ndarray) and k in self.stack_keys:
                    v = torch.from_numpy(v)
                data_dict[k].append(v)

        for key, value in data_dict.items():
            if key in self.stack_keys:
                data_dict[key] = torch.stack(data_dict[key], 0)

        return data_dict
