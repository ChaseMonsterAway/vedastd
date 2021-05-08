#!/usr/bin/env python
# encoding: utf-8
import torch
import torch.nn as nn

from vedastd.models.weight_init import init_weights
from ..utils import build_module
from .registry import HEADS


@HEADS.register_module
class DBHead(nn.Module):

    def __init__(self,
                 k,
                 binary,
                 thresh,
                 out_name=None,
                 adaptive=True,
                 fuse_binary=False):
        super(DBHead, self).__init__()
        self.adaptive = adaptive
        self.fuse_binary = fuse_binary
        self.binarize_name = binary['name']

        binary_layers = []
        for layer in binary['layers']:
            binary_layers.append(build_module(layer))
        self.binarize = nn.Sequential(*binary_layers)

        if thresh:
            assert out_name is not None and isinstance(out_name, str),\
                'You should specify the name of final output'
            self.thresh_name = thresh.pop('name')
            self.out_name = out_name
            thresh_layers = []
            for layer in thresh['layers']:
                thresh_layers.append(build_module(layer))
            self.thresh = nn.Sequential(*thresh_layers)

        self.k = k
        init_weights(self.modules())

    @property
    def with_thresh_layer(self):
        return hasattr(self, 'thresh') and self.thresh is not None

    def forward(self, feature):
        result = {}
        binary = self.binarize(feature)
        result[self.binarize_name] = binary

        if self.adaptive and self.training:
            if self.fuse_binary:
                feature = torch.cat(
                    (feature,
                     nn.functional.interpolate(binary, feature.shape[2:])), 1)
            thresh = self.thresh(feature)
            thresh_binary = self.step_function(binary, thresh)
            result[self.thresh_name] = thresh
            result[self.out_name] = thresh_binary

        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))
