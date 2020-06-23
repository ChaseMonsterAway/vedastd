import torch

from .base_loss import BaseLoss
from .registry import CRITERIA


@CRITERIA.register_module
class MaskL1Loss(BaseLoss):
    def __init__(self, *args, **kwargs):
        super(MaskL1Loss, self).__init__(*args, **kwargs)

    def forward(self, pred, target):
        pred, target, target_mask = self.extract_pairs(pred, target)

        loss = (torch.abs(pred[:, 0] - target) * target_mask).sum() / target_mask.sum()

        return loss
