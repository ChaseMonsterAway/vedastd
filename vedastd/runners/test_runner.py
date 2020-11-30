import torch
import copy

from tqdm import tqdm
import numpy as np

from .inference_runner import InferenceRunner
from ..postpocessor import SearchPostprocessor


class TestRunner(InferenceRunner):
    def __init__(self, test_cfg, inference_cfg, common_cfg):
        assert 'metric' in common_cfg
        super(TestRunner, self).__init__(inference_cfg, common_cfg)

        self.test_dataloader = self._build_dataloader(test_cfg['data'])
        self.search = False
        if 'postprocessor' in test_cfg:
            self.postprocessor = self._build_postprocessor(test_cfg['postprocessor'])
        if isinstance(self.postprocessor, SearchPostprocessor):
            self.metric = [self._build_metric(common_cfg['metric'])
                           for i in range(len(self.postprocessor))]
            self.search = True
            self.logger.info(f'We will measure the performace '
                             f'@{len(self.postprocessor)} postprocessor.'
                             f'It will take some time.')

    def test_batch(self, batch):
        self.model.eval()
        with torch.no_grad():
            img = batch['image']
            if self.use_gpu:
                img = img.cuda()
            pred = self.model(img)
            if self.postprocessor:
                boxes = self.postprocessor(batch, pred, training=False)
                if self.search:
                    for idx, box in enumerate(boxes):
                        self.metric[idx].measure(batch, box, training=False)
                else:
                    self.metric.measure(batch, boxes, training=False)
            else:
                raise ValueError('Post process is Needed to compute the metrics.'
                                 ' Add the postprocess in config file please.')

    def _reset(self):
        if isinstance(self.metric, list):
            for metric in self.metric:
                metric.reset()
        else:
            self.metric.reset()

    def __call__(self):
        self.logger.info('Start testing')
        self._reset()

        for batch in tqdm(self.test_dataloader):
            self.test_batch(batch)
        if self.search:
            collect_res = {}
            self.logger.info("Test:")
            for metric, cfg in zip(self.metric, self.postprocessor.cfgs):
                self.logger.info(f'@cfg {cfg}')
                res = metric.metrics
                self.logger.info(f'{res}')
                if isinstance(res, dict):
                    for key, value in res.items():
                        if key not in collect_res:
                            collect_res[key] = [value]
                        else:
                            collect_res[key].append(value)
                self.logger.info('Statistical results:')
            for key, value in collect_res.items():
                idx = np.argmax(value)
                self.logger.info(f'The cfg @{self.postprocessor.cfgs[idx]} '
                                 f'have the highest {key} value: {value[idx]}')
        else:
            self.logger.info("Test:")
            self.logger.info(f'{(self.metric.metrics)}')
