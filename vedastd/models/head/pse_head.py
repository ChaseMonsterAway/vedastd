#!/usr/bin/env python
# encoding: utf-8
import torch
import torch.nn as nn

from .registry import HEADS
from ..utils import build_module, build_torch_nn


@HEADS.register_module
class PseHead(nn.Module):

    def __init__(self, layers, scale=1, name=None):
        super(PseHead, self).__init__()
        self.scale = scale
        assert name is not None and isinstance(name, tuple), \
            'You should specify the name of final output'
        self.name = name

        binary_layers = []
        for layer in layers:
            binary_layers.append(build_module(layer))
        self.binarize = nn.Sequential(*binary_layers)

    def forward(self, feature):
        result = {}
        binary = self.binarize(feature)
        result[self.name[0]] = binary[:, 0, :, :].unsqueeze(dim=1)  # b * h * w
        result[self.name[1]] = binary[:, 1:, :, :]  # b * c * h * w

        return result
