import cv2
import torch
import numpy as np

from vedastd.criteria import DiceLoss, MultiDiceLoss, build_criterion
from vedastd.datasets import build_datasets
from vedastd.datasets.transforms import build_transform
from vedastd.dataloaders.base import BaseDataloader
from vedastd.utils.config import Config


def main():
    #torch.manual_seed(0)
    cfg = Config.fromfile('./configs/psenet_resnet50.py')
    criterion = build_criterion(cfg['criterion'])[1]
    # pred = dict(
    #     pred_text_map=torch.rand(4, 1, 224, 224),
    # )
    # gt = dict(
    #     text_map=torch.rand(4, 1, 224, 224),
    #     text_mask=torch.rand(4, 1, 224, 224),
    # )
    B =10
    pred = dict(
        pred_text_map=torch.rand(B, 1, 224, 224),
        pred_kernels_map=torch.rand(B, 6, 224, 224),
    )
    gt = dict(
        kernels_map=torch.rand(B, 6, 224, 224),
        text_mask=torch.rand(B, 1, 224, 224),
    )
    loss = criterion(pred, gt)
    print(loss)


if __name__ == '__main__':
    main()
