from abc import abstractmethod

import torch
import torch.nn as nn

from .registry import CRITERIA


@CRITERIA.register_module
class BaseLoss(nn.Module):

    def __init__(self, pred_map: str, gt_map: str, gt_mask: str, loss_weight: (int, float), loss_name: str,
                 ohem: bool = False):
        super(BaseLoss, self).__init__()
        self.pred_map = pred_map
        self.gt_map = gt_map
        self.gt_mask = gt_mask
        self.loss_weight = loss_weight
        self.name = loss_name
        self.ohem = ohem

    def extract_pairs(self, pred, target):
        pred_map = pred[self.pred_map]
        target_map = target[self.gt_map]
        tgt_mask = target[self.gt_mask]
        if isinstance(target_map, list):
            target_map = torch.cat(target_map, 1)
        if isinstance(tgt_mask, list):
            tgt_mask = torch.cat(tgt_mask, 1)

        return pred_map, target_map, tgt_mask

    @abstractmethod
    def forward(self, pred, target):
        pass
