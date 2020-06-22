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
    transforms = build_transform(cfg['transforms'])
    datasets = build_datasets(cfg['dataset'], dict(transforms=transforms))[0]
    dataloader = BaseDataloader(datasets)
    for idx, batch in enumerate(dataloader):
        # for idx in range(len(datasets)):
        #     batch = datasets[idx]
        for key, values in batch.items():
            print(key)
            if isinstance(values, list):
                for v in values:
                    cv2.imshow('%s' % key, tensor_to_img(v))
                    # cv2.waitKey()
            else:
                cv2.imshow('%s' % key, tensor_to_img(values))
        cv2.waitKey()

        print('done')

    print('done')


if __name__ == '__main__':
    main()
