import logging

import torch

from .base_loss import BaseLoss
from .registry import CRITERIA

logger = logging.getLogger('MASK')


@CRITERIA.register_module
class MaskL1Loss(BaseLoss):

    def __init__(self, *args, **kwargs):
        super(MaskL1Loss, self).__init__(*args, **kwargs)

    def forward(self, pred, target):
        pred, target, target_mask = self.extract_pairs(pred, target)
        mask_sum = target_mask.sum()
        if mask_sum.item() == 0:
            return 0
        loss = (torch.abs(pred - target.to(pred.device)) *
                target_mask.to(pred.device)).sum() / mask_sum.to(pred.device)
        return loss * self.loss_weight
