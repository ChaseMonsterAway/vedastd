import argparse
import os
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
    parser.add_argument('--score', action='store_true', default=False,
                        help='Show score or not.')
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

        outputs, aug = runner(batch)
        boxes, scores = outputs
        boxes = boxes[0]
        scores = scores[0]
        for idx, box in enumerate(boxes):
            cv2.polylines(image, [box.reshape(-1, 1, 2).astype(np.int)], True,
                          (0, 255, 0), 2)
            if args.score:
                cv2.putText(image, str(scores[idx])[:6], (int(box[3, 0]), int(box[3, 1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow('g', image)
        key = cv2.waitKey()
        if key == ord('q'):
            break


if __name__ == '__main__':
    main()
