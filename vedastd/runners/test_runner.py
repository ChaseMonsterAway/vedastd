import torch


from .inference_runner import InferenceRunner


class TestRunner(InferenceRunner):
    def __init__(self, test_cfg, inference_cfg, common_cfg=None):
        super(TestRunner, self).__init__(inference_cfg, common_cfg)

        self.test_dataloader = self._build_dataloader(test_cfg['data'])

    def test_batch(self, batch):
        self.model.eval()
        with torch.no_grad():
            img = batch['input']
            if self.use_gpu:
                img = img.cuda()
            pred = self.model(img)
            if self.postprocessor:

                boxes = self.postprocessor(batch, pred, training=False)
                res = self.metric.validate_measure(batch, boxes, training=False)
        return res

    def __call__(self):
        self.logger.info('Start testing')
        self.metric.reset()

        for batch in self.test_dataloader:
            self.test_batch(batch)

        self.logger.info(
            'Test, acc %.4f, edit %s' % (self.metric.avg['acc']['true'],
                                         self.metric.avg['edit']))
