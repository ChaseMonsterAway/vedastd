from vedastd.utils import build_from_cfg
from .registry import HEADS


def build_head(cfg, default_args=None):
    heads = build_from_cfg(cfg, HEADS, default_args)

    return heads
