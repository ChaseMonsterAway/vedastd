import cv2
import torch
import numpy as np

from vedastd.criteria import build_criterion
from vedastd.utils.config import Config


def main():
    # torch.manual_seed(0)
    cfg = Config.fromfile('./configs/psenet_resnet50.py')
    criterion = build_criterion(cfg['criterion'])[0]
    print(criterion)
    B = 1

    pred = dict(
        pred_text_map=torch.rand(B, 1, 224, 224),
    )
    gt = dict(
        text_map=torch.rand(B, 1, 224, 224),
        text_mask=torch.rand(B, 1, 224, 224),
    )

    # pred = dict(
    #     pred_text_map=torch.rand(B, 1, 224, 224),
    #     pred_kernels_map=torch.rand(B, 6, 224, 224),
    # )
    # gt = dict(
    #     kernels_map=torch.rand(B, 6, 224, 224),
    #     text_mask=torch.rand(B, 1, 224, 224),
    # )
    loss = criterion(pred, gt)
    print(loss)


if __name__ == '__main__':
    main()
