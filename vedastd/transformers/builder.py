from vedastd.utils import build_from_cfg

from .registry import TRANSFORMS

import albumentations as alb


def build_transform(cfg):
    tfs = []
    for icfg in cfg:
        if TRANSFORMS.get(icfg.get('type')) is not None:
            tf = build_from_cfg(icfg, TRANSFORMS)
        elif hasattr(alb, icfg.get('type')):
            tf = build_from_cfg(icfg, alb, src='module')
        else:
            raise AttributeError(f"Invalid class {icfg.get('type')}")

        tfs.append(tf)
    aug = alb.Compose(
        tfs, p=1, keypoint_params=alb.KeypointParams(format='xy')
    )

    return aug
