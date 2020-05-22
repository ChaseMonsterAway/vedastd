from vedastd.utils import build_from_cfg
from .registry import SEQUENCE_ENCODERS


def build_sequence_encoder(cfg, default_args=None):
    sequence_encoder = build_from_cfg(cfg, SEQUENCE_ENCODERS, default_args)

    return sequence_encoder
