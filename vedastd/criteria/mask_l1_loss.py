import torch

from .base_loss import BaseLoss
from .registry import CRITERIA


@CRITERIA.register_module
class MaskL1Loss(BaseLoss):
    def __init__(self, *args, **kwargs):
        super(MaskL1Loss, self).__init__(*args, **kwargs)

    def _forward(self, pred, target):
        # pdb.set_trace()
        pred, target, target_mask = self.extract_pairs(pred, target)

        loss = (torch.abs(pred[:, 0] - target.to(pred.device)) * target_mask.to(
            pred.device)).sum() / target_mask.sum().to(pred.device)

        return loss


if __name__ == '__main__':
    import random
    import numpy as np


    def seed(n):
        random.seed(n)
        np.random.seed(n)
        torch.manual_seed(n)
        torch.cuda.manual_seed(n)


    seed(11)
    pred = torch.randn(size=(1, 1, 512, 512))
    gt = torch.randn(size=(1, 1, 512, 512))
    mask = torch.randn(size=(1, 512, 512))
    mask = mask.unsqueeze(0)
    l1_loss = MaskL1Loss(pred_map='1',
                         target='1',
                         loss_weight=1.0,
                         loss_name='2')
    loss = l1_loss({'1': pred}, {'1': torch.cat((gt, mask), 1)})
    print(loss)
