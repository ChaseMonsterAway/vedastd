import pdb
from abc import abstractmethod

import torch
import torch.nn as nn

from .registry import CRITERIA


@CRITERIA.register_module
class BaseLoss(nn.Module):

    def __init__(self, pred_map: str, target: str, loss_weight: (int, float), loss_name: str,
                 ohem: bool = False, effective_mask_first: bool = False):
        super(BaseLoss, self).__init__()
        self.pred_map = pred_map
        self.target = target
        self.effective_mask_first = effective_mask_first
        self.loss_weight = loss_weight
        self.name = loss_name
        self.ohem = ohem

    def extract_pairs(self, pred, target):
        pred_map = pred[self.pred_map]
        target_masks = target[self.target]
        B, C, H, W = target_masks.size()
        if not self.effective_mask_first:
            gt_masks = target_masks[:, :C // 2, :, :]
            effective_masks = target_masks[:, C // 2:, :, :]
        else:
            effective_masks = target_masks[:, :C // 2, :, :]
            gt_masks = target_masks[:, C // 2:, :, :]

        return pred_map, gt_masks, effective_masks

    @abstractmethod
    def forward(self, pred, target):
        pass
