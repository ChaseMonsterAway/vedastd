import numpy as np
import pdb
import torch

from ..models import build_model
from ..utils import load_checkpoint
from .base import Common


class InferenceRunner(Common):

    def __init__(self, inference_cfg, common_cfg=None):
        inference_cfg = inference_cfg.copy()
        common_cfg = {} if common_cfg is None else common_cfg.copy()

        common_cfg['gpu_id'] = inference_cfg.pop('gpu_id')
        super(InferenceRunner, self).__init__(common_cfg)

        # build postprocessor
        if 'postprocessor' in inference_cfg:
            self.postprocessor = self._build_postprocessor(
                inference_cfg['postprocessor'])

        # build test transform
        self.transform = self._build_transform(inference_cfg['transforms'])
        # build model
        self.model = self._build_model(inference_cfg['model'])
        self.model.eval()

    def _build_model(self, cfg):
        self.logger.info('Build model')

        model = build_model(cfg)
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            model.cuda()

        return model

    def load_checkpoint(self, filename, map_location='default', strict=True):
        self.logger.info('Load checkpoint from {}'.format(filename))

        if map_location == 'default':
            if self.use_gpu:
                device_id = torch.cuda.current_device()
                map_location = lambda storage, loc: storage.cuda(device_id)
            else:
                map_location = 'cpu'

        return load_checkpoint(self.model, filename, map_location, strict)

    def __call__(self, batch):
        with torch.no_grad():
            image = batch['image']
            dummy_points = [[790, 302, 903, 304, 902, 335, 790, 335]]
            aug = self.transform(image=image, keypoints=dummy_points)
            image = aug['image'].unsqueeze(0)
            aug['image'] = image
            aug['shape'] = np.array(batch['shape'])

            if self.use_gpu:
                image = image.cuda()
            pred = self.model(image)
            for key, value in aug.items():
                if key != 'image':
                    aug[key] = [aug[key]]
            boxes = self.postprocessor(aug, pred, training=False)

        return boxes, aug
