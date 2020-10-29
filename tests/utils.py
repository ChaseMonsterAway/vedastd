import cv2
import torch
import numpy as np


def imshow(name, img, is_show):
    if is_show:
        cv2.namedWindow(name, 0)
        cv2.imshow(name, img)
    else:
        pass


def show_polygon(image, polygon, tag=None, name=None, is_show=True):
    show_img = image.copy()
    for idx, p in enumerate(polygon):
        if tag is not None:
            color = (0, 255, 0) if tag[idx] else (255, 0, 0)
        else:
            color = (0, 0, 255)

        cv2.polylines(show_img, [p.reshape(-1, 1, 2).astype(np.int)], True, color, 10)
    if name is None:
        name = 'polygon'
    imshow(name, show_img, is_show)


def tensor_to_numpy(inp):
    assert isinstance(inp, torch.Tensor), f'input should be torch.' \
                                          f'Tensor but got {type(inp)}'
    if len(inp.size()) == 3:
        return inp.permute(1, 2, 0).cpu().numpy()
    else:
        return inp.cpu().numpy()
