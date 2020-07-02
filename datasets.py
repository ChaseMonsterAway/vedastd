import cv2
import torch
import numpy as np

from vedastd.datasets import build_datasets
from vedastd.transformers import build_transform
from vedastd.dataloaders.base import BaseDataloader
from vedastd.utils.config import Config


def tensor_to_img(t_img: torch.Tensor):
    t_img = t_img[0]
    if t_img.ndim == 3:
        t_img = t_img.permute(1, 2, 0).cpu().numpy()
    else:
        t_img = t_img.cpu().numpy()
    t_img = (t_img - np.min(t_img)) / (np.max(t_img) - np.min(t_img))
    t_img = (t_img * 255).clip(0, 255).astype(np.uint8)

    return t_img


def main():
    cfg = Config.fromfile('./configs/dummy_config.py')
    train_transforms = [
        dict(type='Flip', p=1),
        dict(type='MakeShrinkMap', ratios=[0.9, 0.8], max_shr=0.9, min_text_size=4),
        dict(type='HueSaturationValue', p=1),
        dict(type='RandomCropBasedOnBox'),
        # dict(type='MaskDropout', p=1),
        # dict(type='RandomCrop', p=1, width=256, height=256),
    ]

    transforms = build_transform(train_transforms)
    datasets = build_datasets(cfg['dataset'], dict(transforms=transforms))
    for i in range(10):
        out = datasets[0][2]
        cv2.imshow('mask', out['masks'][0])
        cv2.imshow('img', out['image'])

        cv2.waitKey()

    print('done')


if __name__ == '__main__':
    main()
