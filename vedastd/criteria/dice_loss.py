from .base_loss import BaseLoss
from .registry import CRITERIA


@CRITERIA.register_module
class DiceLoss(BaseLoss):

    def __init__(self, eps, *args, **kwargs):
        super(DiceLoss, self).__init__(*args, **kwargs)
        self.eps = eps

    def _compute(self, pred, gt, mask, weights=None):
        """
        Args:
            pred:
            gt:
            mask:
            weights: to see the usages

        Returns:
        """
        if pred.dim() == 4:
            pred = pred[:, 0, :, :]
            gt = gt[:, 0, :, :]
            mask = mask[:, 0, :, :]
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask

        intersection = (pred * gt * mask).sum()
        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1

        return loss

    def forward(self, pred, target):
        pmap, tmap, tmask = self.extract_pairs(pred, target)
        loss = self._compute(pmap, tmap, tmask)

        return loss * self.loss_weight
