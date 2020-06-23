import cv2
import torch
import numpy as np

from vedastd.datasets import build_datasets
from vedastd.datasets.transforms import build_transform
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
    transforms = build_transform(cfg['train_transforms'])
    datasets = build_datasets(cfg['dataset'], dict(transforms=transforms))
    dataloader = BaseDataloader(datasets)
    for batch in dataloader:
        img = tensor_to_img(batch['input'])
        poly = [p.numpy() for p in batch['polygon']]
        smp = tensor_to_img(batch['seg_map'])
        smk = tensor_to_img(batch['seg_mask'])
        cv2.imshow('mp', smp[0, 0])
        cv2.imshow('mk', smk[0, 0])
        try:
            ratio = batch['ratio'].item()
        except:
            ratio = 1
        h, w = img.shape[:2]
        new_img = cv2.resize(img, (int(w / ratio), int(h / ratio)))
        # new_img = img.copy()
        for p in poly:
            # p = (p * ratio).astype(int)
            cv2.rectangle(new_img, (p[0, 0, 0], p[0, 0, 1]),
                          (p[0, 2, 0], p[0, 2, 1]), (255, 0, 255))
        cv2.imshow('nimg', new_img)
        cv2.waitKey()

        # for idx in range(len(datasets)):
        #     batch = datasets[idx]
        # for key, values in batch.items():
        #     print(key)
        #     if isinstance(values, list):
        #         for v in values:
        #             cv2.imshow('%s' % key, tensor_to_img(v))
        #     else:
        #         cv2.imshow('%s' % key, tensor_to_img(values))
        # cv2.waitKey()

        print('done')

    print('done')


if __name__ == '__main__':
    main()
