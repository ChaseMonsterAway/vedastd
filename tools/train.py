import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))

from vedastd.runner import TrainRunner
from vedastd.utils import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a scene text recognition model')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    _, fullname = os.path.split(cfg_path)
    fname, ext = os.path.splitext(fullname)

    root_workdir = cfg.pop('root_workdir')
    workdir = os.path.join(root_workdir, fname)
    os.makedirs(workdir, exist_ok=True)

    train_cfg = cfg['train']
    inference_cfg = cfg['inference']
    common_cfg = cfg['common']
    common_cfg['workdir'] = workdir

    runner = TrainRunner(train_cfg, inference_cfg, common_cfg)
    runner()


if __name__ == '__main__':
    main()
