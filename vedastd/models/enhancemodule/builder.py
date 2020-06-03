from vedastd.utils import build_from_cfg
from .registry import ENHANCE


def build_enhance(cfg, default_args=None):
    decoder = build_from_cfg(cfg, ENHANCE, default_args)
    return decoder
