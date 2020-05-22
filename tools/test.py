import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../vedastr'))

from vedastd.assembler import assemble


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a scene text recognition model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_fp = args.config
    checkpoint = args.checkpoint

    runner = assemble(cfg_fp, test_mode=True, checkpoint=checkpoint)
    runner()


if __name__ == '__main__':
    main()
