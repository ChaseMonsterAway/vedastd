import numpy as np
import torch
from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours
from tqdm import tqdm

from ..postpocessor import SearchPostprocessor
from .inference_runner import InferenceRunner


class TestRunner(InferenceRunner):

    def __init__(self, test_cfg, inference_cfg, common_cfg):
        assert 'metric' in common_cfg
        super(TestRunner, self).__init__(inference_cfg, common_cfg)

        self.test_dataloader = self._build_dataloader(test_cfg['data'])
        self.search = False
        if 'postprocessor' in test_cfg:
            self.postprocessor = self._build_postprocessor(
                test_cfg['postprocessor'])
        if 'search_params' in test_cfg:
            self.params = test_cfg['search_params']

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
                raise ValueError(
                    'Post process is Needed to compute the metrics.'
                    ' Add the postprocess in config file please.')

    def test_all_data(self, **kwargs):
        self.metric.reset()
        self.postprocessor.set_params(**kwargs)
        for batch in tqdm(self.test_dataloader):
            self.test_batch(batch)
        return self.metric.metrics['fmeasure']

    def _reset(self):
        if isinstance(self.metric, list):
            for metric in self.metric:
                metric.reset()
        else:
            self.metric.reset()

    def _optimizer(self, n_iter=10):
        bayesian_opt = BayesianOptimization(
            f=self.test_all_data,
            pbounds=self.params,
            random_state=1234,
            verbose=2)
        bayesian_opt.maximize(n_iter=n_iter)
        self.logger.info('Final result:', bayesian_opt.max)

    def _grid(self):
        for i in range(30, 70, 5):
            for j in range(i, 75, 5):
                self.test_all_data(thresh=i / 100.0, box_thresh=j / 100.0)
                self.logger.info(
                    f"thresh {i} box_thresh {j} fmeasure {self.metric.metrics['fmeasure']}, "
                    f"recall {self.metric.metrics['recall']}, precision {self.metric.metrics['precision']}"
                )

    def __call__(self):
        self.logger.info('Start testing')
        self._reset()
        # self._optimizer()
        self._grid()
        #
        # for batch in tqdm(self.test_dataloader):
        #     self.test_batch(batch)
        # if self.search:
        #     collect_res = {}
        #     self.logger.info("Test:")
        #     for metric, cfg in zip(self.metric, self.postprocessor.cfgs):
        #         self.logger.info(f'@cfg {cfg}')
        #         res = metric.metrics
        #         self.logger.info(f'{res}')
        #         if isinstance(res, dict):
        #             for key, value in res.items():
        #                 if key not in collect_res:
        #                     collect_res[key] = [value]
        #                 else:
        #                     collect_res[key].append(value)
        #         self.logger.info('Statistical results:')
        #     for key, value in collect_res.items():
        #         idx = np.argmax(value)
        #         self.logger.info(f'The cfg @{self.postprocessor.cfgs[idx]} '
        #                          f'have the highest {key} value: {value[idx]}')
        # else:
        #     self.logger.info("Test:")
        #     self.logger.info(f'{(self.metric.metrics)}')
