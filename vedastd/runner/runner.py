import logging
import os.path as osp
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from vedastd.utils.checkpoint import load_checkpoint, save_checkpoint
from .registry import RUNNERS
from ..utils.metrics import QuadMeasurer

np.set_printoptions(precision=4)

logger = logging.getLogger()


@RUNNERS.register_module
class Runner(object):
    """ Runner

    """

    def __init__(self,
                 epochs,
                 loader,
                 model,
                 criterion,
                 optim,
                 lr_scheduler,
                 postprocessor,
                 iterations,
                 workdir,
                 metric=None,
                 start_iters=0,
                 trainval_ratio=1,
                 snapshot_interval=1,
                 gpu=True,
                 test_cfg=None,
                 test_mode=False,
                 grad_clip=0):
        self.epochs = epochs
        self.loader = loader
        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.start_iters = start_iters
        self.iterations = iterations
        self.postprocessor = postprocessor
        self.workdir = workdir
        self.trainval_ratio = trainval_ratio
        self.snapshot_interval = snapshot_interval
        self.gpu = gpu
        self.test_cfg = test_cfg
        self.test_mode = test_mode
        self.grad_clip = grad_clip
        self.best_norm = 0
        self.best_acc = 0
        self.c_iter = 0

    def __call__(self):
        if self.test_mode:
            self.test_epoch()
        else:
            logger.info('Start train...')
            for epoch in range(self.epochs):
                for iters, batch in enumerate(self.loader['train']):
                    self.c_iter += 1
                    self.train_batch(batch)
                    if self.lr_scheduler:
                        self.lr_scheduler.step()
                    # if self.trainval_ratio > 0 \
                    #         and (iters + 1) % self.trainval_ratio == 0 \
                    #         and self.loader.get('val'):
                if epoch % 10 == 0:
                    self.validate_epoch()
                    # self.metric.reset()
                    # if (iters + 1) % self.snapshot_interval == 0:
                    #     self.save_model(out_dir=self.workdir,
                    #                     filename=f'iter{iters + 1}.pth',
                    #                     iteration=iters,
                    #                     )

    def validate_epoch(self):
        logger.info('Iteration %d, Start validating' % self.c_iter)
        for batch in self.loader['val']:
            self.validate_batch(batch)

    def test_epoch(self):
        logger.info('Start testing')
        logger.info('test info: %s' % self.test_cfg)
        self.metric.reset()
        for img, label in self.loader['test']:
            self.test_batch(img, label)

        logger.info('Test, acc %.4f, edit %s' % (self.metric.avg['acc']['true'],
                                                 self.metric.avg['edit']))

    def train_batch(self, batch):
        self.model.train()
        self.optim.zero_grad()
        img = batch['input']
        if self.gpu:
            img = img.cuda()
        pred = self.model(img)

        loss_infos = {criterion.name: criterion(pred, batch) for criterion in self.criterion}
        loss = torch.stack(list(loss_infos.values()), 0).sum()

        loss.backward()
        self.optim.step()

        if self.c_iter % 10 == 0:
            if self.postprocessor:
                boxes = self.postprocessor(batch, pred)
                res = QuadMeasurer().validate_measure(batch, boxes)
                logger.info(f'{res}')

            logger.info(f'Train, Iter {self.c_iter}, LR {self.lr} loss {loss.item()}')

            for key, value in loss_infos.items():
                logger.info(f'{key}:\t{value}')

    def validate_batch(self, batch):
        self.model.eval()
        with torch.no_grad():
            img = batch['input']
            if self.gpu:
                img = img.cuda()
            pred = self.model(img)
            if self.postprocessor:
                boxes = self.postprocessor(batch, pred)
                res = QuadMeasurer().validate_measure(batch, boxes)
                logger.info(f'EVAL!!! \n {res}')

    def test_batch(self, img, label):
        self.model.eval()
        with torch.no_grad():
            label_input, label_length, label_target = self.converter.test_encode(label)
            if self.gpu:
                img = img.cuda()
                label_input = label_input.cuda()

            if self.need_text:
                pred = self.model(img, label_input)
            else:
                pred = self.model(img)
            preds_prob = F.softmax(pred, dim=2)
            preds_prob, pred_index = preds_prob.max(dim=2)
            pred_str = self.converter.decode(pred_index)

            self.metric.measure(pred_str, label, preds_prob)

    def save_model(self,
                   out_dir,
                   filename,
                   iteration,
                   save_optimizer=True,
                   meta=None):
        if meta is None:
            meta = dict(iter=iteration + 1, lr=self.lr, iters=self.iterations)
        else:
            meta.update(iter=iteration + 1, lr=self.lr, iters=self.iterations)

        filepath = osp.join(out_dir, filename)
        optimizer = self.optim if save_optimizer else None
        logger.info('Save checkpoint %s', filename)
        save_checkpoint(self.model,
                        filepath,
                        optimizer=optimizer,
                        meta=meta)

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        logger.info('Resume from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict,
                               logger)

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
        lr = [x['lr'] for x in self.optim.param_groups]
        return np.array(lr)

    @lr.setter
    def lr(self, val):
        for idx, param in enumerate(self.optim.param_groups):
            if isinstance(val, Iterable):
                param['lr'] = val[idx]
            else:
                param['lr'] = val

    def resume(self,
               checkpoint,
               resume_lr=True,
               resume_iters=True,
               resume_optimizer=False,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(checkpoint, map_location=map_location)
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optim.load_state_dict(checkpoint['optimizer'])
        if 'meta' in checkpoint and resume_iters:
            self.iterations = checkpoint['meta']['iters']
            self.start_iters = checkpoint['meta']['iter']
            self.iter = checkpoint['meta']['iter']
            self.c_iter = self.start_iters + 1
        if 'meta' in checkpoint and resume_lr:
            self.lr = checkpoint['meta']['lr']
