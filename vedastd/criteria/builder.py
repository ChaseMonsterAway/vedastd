from vedastd.utils import build_from_cfg

from .registry import CRITERIA


def build_criterion(cfgs):

    criterion_list = []
    for cfg in cfgs:
        criterion = build_from_cfg(cfg, CRITERIA, mode='registry')
        criterion_list.append(criterion)

    return criterion_list
