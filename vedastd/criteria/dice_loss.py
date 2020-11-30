import torch

from .base_loss import BaseLoss
from .registry import CRITERIA
from .utils import ohem_batch


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

    def _forward(self, pred, target):  # only pred is in CUDA
        pmap, tmap, tmask = self.extract_pairs(pred, target)
        if self.ohem:
            tmask = ohem_batch(pmap, tmap, tmask)
        tmap, tmask = tmap.to(pmap.device), tmask.to(pmap.device)
        loss = self._compute(pmap, tmap, tmask)

        return loss


@CRITERIA.register_module
class MultiDiceLoss(BaseLoss):

    def __init__(self, eps, score_map, *args, **kwargs):
        super(MultiDiceLoss, self).__init__(*args, **kwargs)
        self.eps = eps
        self.score_map = score_map

    @staticmethod
    def mask_process(score_map, gt_mask):
        # assert score_map.shape == gt_mask.shape
        if score_map.dim() == 4:
            score_map = score_map[:, 0, :, :]
            gt_mask = gt_mask[:, 0, :, :]
        mask0 = torch.sigmoid(score_map).data.cpu()
        mask1 = gt_mask.data.cpu()
        selected_masks = (mask0 > 0.5) & (mask1 > 0.5)
        return selected_masks.float()

    def _compute(self, pred, gt, mask, weights=None):
        """
        Args:
            pred:
            gt:
            mask:
            weights: to see the usages

        Returns:
        """
        assert pred.shape == gt.shape
        assert pred.shape != mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask

        loss_kernels = []
        for i in range(pred.shape[1]):
            pred_i = torch.sigmoid(pred[:, i, :, :])
            gt_i = gt[:, i, :, :]
            intersection = (pred_i * gt_i * mask).sum()
            union = (pred_i * mask).sum() + (gt_i * mask).sum() + self.eps
            loss = 1 - 2.0 * intersection / union
            # assert loss <= 1
            loss_kernels.append(loss)
        return sum(loss_kernels) / len(loss_kernels)

    def _forward(self, pred, target):
        # score_map shape b*1*w*h
        score_map = pred[self.score_map]
        # pmap,tmap.shape == B*6*wh, tmask shape == b*1*wh
        pmap, tmap, tmask = self.extract_pairs(pred, target)
        selected_masks = self.mask_process(score_map, tmask)
        tmap, selected_masks = tmap.to(pmap.device), selected_masks.to(pmap.device)
        loss = self._compute(pmap, tmap, selected_masks)

        return loss


if __name__ == '__main__':
    import random
    import numpy as np


    def seed(n):
        random.seed(n)
        np.random.seed(n)
        torch.manual_seed(n)
        torch.cuda.manual_seed(n)


    seed(0)
    pred = torch.randn(size=(1, 1, 512, 512)).sigmoid()
    gt = torch.randn(size=(1, 1, 512, 512)).sigmoid().round()
    mask = torch.randn(size=(1, 512, 512)).sigmoid().round()
    mask = mask.unsqueeze(0)
    l1_loss = DiceLoss(pred_map='1',
                       target='1',
                       loss_weight=1.0,
                       loss_name='2',
                       eps=1e-6)
    loss = l1_loss({'1': pred}, {'1': torch.cat((gt, mask), 1)})
    print(loss)
