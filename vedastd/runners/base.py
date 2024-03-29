import numpy as np
import os
import pdb
import random
import torch
from torch.backends import cudnn

from ..dataloaders import build_dataloader
from ..dataloaders.collate_fn import build_collate_fn
from ..datasets import build_datasets
from ..logger import build_logger
from ..metrics import build_metric
from ..postpocessor import build_postprocessor
from ..transforms import build_transform


class Common(object):

    def __init__(self, cfg):
        super(Common, self).__init__()

        # build logger
        logger_cfg = cfg.get('logger')
        if logger_cfg is None:
            logger_cfg = dict(
                handlers=(dict(type='StreamHandler', level='INFO'), ))
        self.workdir = cfg.get('workdir')
        self.logger = self._build_logger(logger_cfg)

        # set gpu devices
        self.use_gpu = self._set_device(cfg.get('gpu_id', ''))

        # set cudnn configuration
        self._set_cudnn(
            cfg.get('cudnn_deterministic', False),
            cfg.get('cudnn_benchmark', False))

        # set seed
        self._set_seed(cfg.get('seed'))

        # build postprocessor
        if 'postprocessor' in cfg:
            self.postprocessor = self._build_postprocessor(
                cfg['postprocessor'])

        # build metric
        if 'metric' in cfg:
            self.metric = self._build_metric(cfg['metric'])

    def _build_logger(self, cfg):
        return build_logger(cfg, dict(workdir=self.workdir))

    def _set_device(self, gpu_id):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        if torch.cuda.is_available():
            self.logger.info('Use GPU {}'.format(gpu_id))
            use_gpu = True
        else:
            self.logger.info('Use CPU')
            use_gpu = False

        return use_gpu

    def _set_seed(self, seed):
        if seed:
            self.logger.info('Set seed {}'.format(seed))
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _set_cudnn(self, deterministic, benchmark):
        self.logger.info('Set cudnn deterministic {}'.format(deterministic))
        cudnn.deterministic = deterministic

        self.logger.info('Set cudnn benchmark {}'.format(benchmark))
        cudnn.benchmark = benchmark

    def _build_metric(self, cfg):
        return build_metric(cfg)

    def _build_transform(self, cfg):
        return build_transform(cfg)

    def _build_postprocessor(self, cfg):
        return build_postprocessor(cfg)

    def _build_dataloader(self, cfg):
        transform = build_transform(cfg['transforms'])
        dataset = build_datasets(cfg['dataset'], dict(transforms=transform))
        collate_fn = build_collate_fn(
            cfg['collate_fn']) if cfg.get('collate_fn') else None
        dataloader = build_dataloader(
            cfg['dataloader'], dict(dataset=dataset, collate_fn=collate_fn))

        return dataloader
