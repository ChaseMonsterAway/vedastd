from .registry import COLLECT_FN

from vedastd.utils import build_from_cfg


def build_collect_fn(cfg, default_args=None):
    collect_fn = build_from_cfg(cfg, COLLECT_FN, default_args)

    return collect_fn
