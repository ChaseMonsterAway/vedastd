import time
import cv2
import torch
import numpy as np

from vedastd.postpocessor import build_postprocessor
from vedastd.utils.config import Config


def main():
    cfg = Config.fromfile('./configs/psenet_resnet50.py')
    post = build_postprocessor(cfg['postprocessor'])
    raw_img = cv2.imread('raw_rail_27.jpg')
    gt = cv2.imread('rail_27.jpg', flags=cv2.IMREAD_GRAYSCALE)
    gt = gt[np.newaxis, np.newaxis, :]

    batch = dict(
        input=torch.unsqueeze(torch.from_numpy(raw_img).permute(2, 0, 1), dim=0),
        shape=torch.unsqueeze(torch.from_numpy(np.array([1260, 160])), dim=0),
        ratio=torch.from_numpy(np.array([1.0])),
    )
    _pred = dict(
        pred_text_map=torch.from_numpy(gt).float(),
        pred_kernels_map=torch.from_numpy(gt).repeat(1, 6, 1, 1).float(),
    )

    start = time.time()
    boxes_batch, scores_batch = post(batch, _pred)
    for boxes, scores in zip(boxes_batch[0], scores_batch[0]):
        cv2.drawContours(raw_img, [np.array(boxes).reshape(4, 2)], -1, (0, 255, 0), 2)
    end = time.time()
    print(end-start)
    cv2.imshow('raw_img', raw_img)
    cv2.waitKey()


if __name__ == '__main__':
    main()
