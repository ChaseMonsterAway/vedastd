from ..utils import build_from_cfg
from .registry import POSTPROCESS


def build_postprocessor(cfg, default_args=None):
    model = build_from_cfg(cfg, POSTPROCESS, default_args)

    return model
