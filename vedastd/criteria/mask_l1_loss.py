import pdb
import torch

from .base_loss import BaseLoss
from .registry import CRITERIA


@CRITERIA.register_module
class MaskL1Loss(BaseLoss):
    def __init__(self, *args, **kwargs):
        super(MaskL1Loss, self).__init__(*args, **kwargs)

    def forward(self, pred, target):
        # pdb.set_trace()
        pred, target, target_mask = self.extract_pairs(pred, target)

        loss = (torch.abs(pred[:, 0] - target.to(pred.device)) * target_mask.to(
            pred.device)).sum() / target_mask.sum().to(pred.device)

        return loss
