from vedastd.utils import build_from_cfg
from .registry import DATALOADERS


def build_dataloader(cfg, default_args=None):
    dataloader = build_from_cfg(cfg, DATALOADERS, default_args)

    return dataloader
