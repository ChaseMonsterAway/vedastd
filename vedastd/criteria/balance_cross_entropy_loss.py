import torch
import torch.nn as nn

from .base_loss import BaseLoss
from .registry import CRITERIA


@CRITERIA.register_module
class BalanceCrossEntropyLoss(BaseLoss):

    def __init__(self, negative_ratio=3.0, eps=1e-6, *args, **kwargs):
        super(BalanceCrossEntropyLoss, self).__init__(*args, **kwargs)
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self, pred: dict, target: dict):
        pred, target, mask = self.extract_pairs(pred, target)
        target = target.to(pred.device)
        mask = mask.to(pred.device)
        positive = (target * mask).byte()
        negative = ((1 - target) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(
            int(negative.float().sum()),
            int(positive_count * self.negative_ratio))
        loss = nn.functional.binary_cross_entropy(
            pred, target.to(pred.device), reduction='none')
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / \
                       (positive_count + negative_count + self.eps)

        return balance_loss * self.loss_weight
