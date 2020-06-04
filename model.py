import cv2
import torch
import numpy as np

from vedastd.datasets import build_datasets
from vedastd.datasets.transforms import build_transform
from vedastd.utils.config import Config
from vedastd.models.builder import build_model


def tensor_to_img(t_img: torch.Tensor):
    if t_img.ndim == 3:
        t_img = t_img.permute(1, 2, 0).cpu().numpy()
    else:
        t_img = t_img.cpu().numpy()
    t_img = (t_img - np.min(t_img)) / (np.max(t_img) - np.min(t_img))
    t_img = (t_img * 255).clip(0, 255).astype(np.uint8)

    return t_img


def main():
    cfg = Config.fromfile('./configs/dummy_config.py')
    model = build_model(cfg['model'])
    dummy_input = torch.randn((1, 3, 224, 224))
    out = model(dummy_input)

    print('done')


if __name__ == '__main__':
    main()
