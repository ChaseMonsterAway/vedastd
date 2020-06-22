from abc import abstractmethod
import torch.nn as nn

from .registry import CRITERIA


@CRITERIA.register_module
class BaseLoss(nn.Module):

    def __init__(self, pred_map: str, gt_map: str, gt_mask: str, loss_weight: (int, float)):
        super(BaseLoss, self).__init__()
        self.pred_map = pred_map
        self.gt_map = gt_map
        self.gt_mask = gt_mask
        self.loss_weight = loss_weight

    def extract_pairs(self, pred, target):
        pred_map = pred[self.pred_map]
        target_map = target[self.gt_map]
        tgt_mask = target[self.gt_mask]

        return pred_map, target_map, tgt_mask

    @abstractmethod
    def forward(self, pred, target):
        pass
