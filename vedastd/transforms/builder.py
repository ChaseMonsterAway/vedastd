import albumentations as alb

from vedastd.utils import build_from_cfg
from .registry import TRANSFORMS
from .transforms import CV2_MODE, CV2_BORDER_MODE


def build_transform(cfg):
    tfs = []

    for icfg in cfg:
        if TRANSFORMS.get(icfg.get('type')) is not None:
            tf = build_from_cfg(icfg, TRANSFORMS)
        elif hasattr(alb, icfg.get('type')):
            if icfg.get('interpolation') and CV2_MODE.get(icfg.get('interpolation')):
                icfg['interpolation'] = CV2_MODE[icfg.get('interpolation')]
            if icfg.get('border_mode') and CV2_BORDER_MODE.get('border_mode'):
                icfg['border_mode'] = CV2_BORDER_MODE[icfg.get('border_mode')]
            tf = build_from_cfg(icfg, alb, mode='module')

        else:
            raise AttributeError(f"Invalid class {icfg.get('type')}")

        tfs.append(tf)

    aug = alb.Compose(
        transforms=tfs, p=1, keypoint_params=alb.KeypointParams(format='xy', remove_invisible=False),
    )

    return aug
