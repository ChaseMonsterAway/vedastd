import os
import pdb
from collections import OrderedDict
from collections.abc import Iterable

import torch

from .inference_runner import InferenceRunner
from ..criteria import build_criterion
from ..lr_schedulers import build_lr_scheduler
from ..optimizers import build_optimizer
from ..utils import save_checkpoint


class TrainRunner(InferenceRunner):
    def __init__(self, train_cfg, inference_cfg, common_cfg=None):
        super(TrainRunner, self).__init__(inference_cfg, common_cfg)

        self.train_dataloader = self._build_dataloader(
            train_cfg['data']['train'])

        if 'val' in train_cfg['data']:
            self.val_dataloader = self._build_dataloader(
                train_cfg['data']['val'])
        else:
            self.val_dataloader = None

        if 'postprocessor' in train_cfg:
            self.postprocessor = self._build_postprocessor(train_cfg['postprocessor'])
        self.max_iterations = train_cfg.get('max_iterations', False)
        self.max_epochs = train_cfg.get('max_epochs', False)
        assert self.max_epochs ^ self.max_iterations, \
            'max_epochs and max_iterations are mutual exclusion'
        if not self.max_iterations:
            self.max_iterations = len(self.train_dataloader) * self.max_epochs
        if not self.max_epochs:
            self.max_epochs = self.max_iterations // len(self.train_dataloader)

        self.optimizer = self._build_optimizer(train_cfg['optimizer'])
        self.criterion = self._build_criterion(train_cfg['criterion'])
        self.lr_scheduler = self._build_lr_scheduler(train_cfg['lr_scheduler'])

        self.log_interval = train_cfg.get('log_interval', 10)
        self.trainval_ratio = train_cfg.get('trainval_ratio', -1)
        self.snapshot_interval = train_cfg.get('snapshot_interval', -1)
        self.grad_clip = train_cfg.get('grad_clip', 5)
        self.save_best = train_cfg.get('save_best', True)

        self.iter = 0

        assert self.workdir is not None
        assert self.log_interval > 0

        self.best = OrderedDict()

        if train_cfg.get('resume'):
            self.resume(**train_cfg['resume'])

    def _build_optimizer(self, cfg):
        return build_optimizer(cfg, dict(params=self.model.parameters()))

    def _build_criterion(self, cfg):
        return build_criterion(cfg)

    def _build_lr_scheduler(self, cfg):
        return build_lr_scheduler(cfg, dict(optimizer=self.optimizer,
                                            niter_per_epoch=len(self.train_dataloader),
                                            max_epochs=self.max_epochs))

    def _validate_epoch(self):
        self.logger.info('Iteration %d, Start validating' % self.iter)
        self.metric.reset()
        for batch in self.val_dataloader:
            self._validate_batch(batch)
        self.logger.info(f'Evaluation \n {self.metric.metrics}')

    def _train_batch(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        img = batch['image']
        if self.use_gpu:
            img = img.cuda()
        pred = self.model(img)

        loss_infos = {criterion.name: criterion(pred, batch) for criterion in
                      self.criterion}
        loss = torch.stack(list(loss_infos.values()), 0).sum()

        loss.backward()
        self.optimizer.step()

        if self.iter % self.log_interval == 0:
            with torch.no_grad():
                boxes = self.postprocessor(batch, pred, training=True)
            if 'train' in self.metric.phase:
                self.metric.measure(batch, boxes, training=True)
                self.logger.info(f'{self.metric.metrics}')
            self.logger.info(
                f'Train, epoch: {self.epoch}, Iter {self.iter}, LR {self.lr} loss {loss.item()}')
            for key, value in loss_infos.items():
                self.logger.info(f'{key}:\t{value}')

    def _validate_batch(self, batch):
        self.model.eval()
        with torch.no_grad():
            img = batch['image']
            if self.use_gpu:
                img = img.cuda()
            pred = self.model(img)
            boxes = self.postprocessor(batch, pred, training=False)
            self.metric.measure(batch, boxes, training=False)

    def __call__(self):
        self.logger.info('Start train...')

        iter_based = self.lr_scheduler._iter_based
        warmup_iters = self.lr_scheduler.warmup_iters

        flag = True
        while flag:
            self.metric.reset()
            for iters, batch in enumerate(self.train_dataloader):
                self._train_batch(batch)
                self.lr_scheduler.iter_nums()  # update steps
                if iter_based:
                    self.lr_scheduler.step()
                elif warmup_iters > 0 and warmup_iters >= self.iter:
                    self.lr_scheduler.step()
                if self.trainval_ratio > 0 \
                        and (self.iter + 1) % self.trainval_ratio == 0 \
                        and self.val_dataloader:
                    self._validate_epoch()
                    self.metric.reset()
                if (self.iter + 1) % self.snapshot_interval == 0:
                    self.save_checkpoint(dir_=self.workdir,
                                         filename=f'iter{self.iter + 1}.pth', )
                if self.iter >= self.max_iterations:
                    flag = False
                    break
            if not iter_based:
                self.lr_scheduler.step()

    @property
    def epoch(self):
        return self.lr_scheduler.last_epoch

    @epoch.setter
    def epoch(self, val):
        """int: Current epoch."""
        self.lr_scheduler.last_epoch = val

    @property
    def iter(self):
        """int: Current iteration."""
        return self.lr_scheduler.last_iter

    @iter.setter
    def iter(self, val):
        """int: Current epoch."""
        self.lr_scheduler.last_iter = val

    @property
    def lr(self):
        return [x['lr'] for x in self.optimizer.param_groups]

    @lr.setter
    def lr(self, val):
        for idx, param in enumerate(self.optimizer.param_groups):
            if isinstance(val, Iterable):
                param['lr'] = val[idx]
            else:
                param['lr'] = val

    def save_checkpoint(self, dir_, filename, save_optimizer=True,
                        save_lr_scheduler=True, meta=None):
        optimizer = self.optimizer if save_optimizer else None
        lr_scheduler = self.lr_scheduler if save_lr_scheduler else None

        filepath = os.path.join(dir_, filename)
        self.logger.info('Save checkpoint {}'.format(filename))
        if meta is None:
            meta = dict(epoch=self.epoch, iter=self.iter, lr=self.lr)
        else:
            meta.update(epoch=self.epoch, iter=self.iter, lr=self.lr)
        save_checkpoint(self.model, filepath, optimizer, lr_scheduler, meta)

    def resume(self, checkpoint, resume_optimizer=False,
               resume_lr_scheduler=False, resume_meta=False,
               map_location='default'):
        checkpoint = self.load_checkpoint(checkpoint,
                                          map_location=map_location)

        if resume_optimizer and 'optimizer' in checkpoint:
            self.logger.info('Resume optimizer')
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if resume_lr_scheduler and 'lr_scheduler' in checkpoint:
            self.logger.info('Resume lr scheduler')
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if resume_meta and 'meta' in checkpoint:
            self.logger.info('Resume meta data')
            self.best = checkpoint['meta']['best']
            self.epoch = checkpoint['meta']['epoch']
            self.iter = checkpoint['meta']['iter']
            self.lr = checkpoint['meta']['lr']
