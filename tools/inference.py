import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../vedastr'))

import cv2

from vedastd.utils import Config
from vedastd.utils.checkpoint import load_checkpoint
from vedastd.datasets.transforms import build_transform
from vedastd.models.builder import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a scene text recognition model')
    parser.add_argument('--config', default='configs/psenet_resnet50.py', help='train config file path')
    parser.add_argument('--checkpoint', default='workdir/psenet_resnet50/iter5.pth', help='checkpoint file path')
    #parser.add_argument('image', type=str, help='input image path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    model_dict = cfg['model']
    transform_dict = cfg['data']['val']['transforms']

    model = build_model(model_dict)
    model.cuda()
    load_checkpoint(model, args.checkpoint)
    transforms = build_transform(transform_dict)
    print(model)
    print(transforms)
    #image = cv2.imread(args.image)
    #shape = image.shape


if __name__ == '__main__':
    main()
