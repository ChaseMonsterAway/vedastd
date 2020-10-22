from collections import OrderedDict

import numpy as np
import torch

from .registry import COLLATE_FN


@COLLATE_FN.register_module
class BaseCollate:

    def __call__(self, batch):
        data_dict = OrderedDict()
        for sample in batch:
            for k, v in sample.items():
                if k not in data_dict:
                    data_dict[k] = []
                if isinstance(v, np.ndarray):
                    v = torch.from_numpy(v)
                data_dict[k].append(v)
        data_dict['image'] = torch.stack(data_dict['image'], 0)

        return data_dict
