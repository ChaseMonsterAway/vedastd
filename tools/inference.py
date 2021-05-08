import argparse
import os
import pdb
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import cv2  # noqa 402
import numpy as np  # noqa 402

from vedastd.runners import InferenceRunner  # noqa 402
from vedastd.utils import Config  # noqa 402


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
        images = [
            os.path.join(args.image, name) for name in os.listdir(args.image)
        ]
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
            nbox = np.array(box).astype(np.int)
            crop_patch = image[np.min(nbox[:, 1]):np.max(nbox[:, 1]), np.min(nbox[:, 0]):np.max(nbox[:, 0]), :]
            cv2.imwrite(r'D:\DATA_ALL\SMT\sm2\res\%s_%s' % (idx, os.path.basename(img)), crop_patch)
            print(box)
            cv2.polylines(image, [box.reshape(-1, 1, 2).astype(np.int)], True,
                          (0, 255, 0), 5)
            # cv2.putText(image,
            #             str(scores[idx])[:6], (int(box[3, 0]), int(box[3, 1])),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.namedWindow('g', 0)
        cv2.imshow('g', image)
        cv2.waitKey()


if __name__ == '__main__':
    main()
