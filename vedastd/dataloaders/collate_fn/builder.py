from .registry import COLLATE_FN

from vedastd.utils import build_from_cfg


def build_collate_fn(cfg, default_args=None):
    collate_fn = build_from_cfg(cfg, COLLATE_FN, default_args)

    return collate_fn
