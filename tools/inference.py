import argparse
import os
import pdb
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import cv2
import numpy as np

from vedastd.runners import InferenceRunner
from vedastd.utils import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('config', type=str, help='Config file path')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file path')
    parser.add_argument('image', type=str, help='input image path')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    deploy_cfg = cfg['deploy']
    common_cfg = cfg.get('common')

    runner = InferenceRunner(deploy_cfg, common_cfg)
    runner.load_checkpoint(args.checkpoint)
    if os.path.isfile(args.image):
        images = [args.image]
    else:
        images = [os.path.join(args.image, name)
                  for name in os.listdir(args.image)]
    for img in images:
        batch = {}
        image = cv2.imread(img)
        batch['image'] = image
        batch['shape'] = np.array(image.shape[:2])

        boxes = runner(batch)
        for box in boxes:
            for b in box:
                print(b)
                print(type(b))
                cv2.rectangle(image, (367, 365),
                              (495, 240), (0, 255, 0), 2)
        cv2.imshow('g', image)
        cv2.waitKey()
        pdb.set_trace()


if __name__ == '__main__':
    main()
