import numpy as np

from .registry import METRICS
from ..utils.icdar15 import DetectionIoUEvaluator


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return self


@METRICS.register_module
class QuadMeasurer2:

    def __init__(self, polygon=False):
        self.recalls = []
        self.precisions = []
        self.fmean = []
        self.polygon = polygon
        self.evaluator = DetectionIoUEvaluator()

    def reset(self):
        self.recalls = []
        self.precisions = []
        self.fmean = []

    def measure(self, batch, boxes, training=False):
        '''
        batch: (image, polygons, ignore_tags
        batch: a dict produced by dataloaders.
            image: tensor of shape (N, C, H, W).
            polygons: tensor of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: tensor of shape (N, K), indicates whether a region is ignorable or not.
            shape: the original shape of images.
            filename: the original filenames of images.
        output: (polygons, ...)
        '''
        results = []
        if training:
            gt_polyons_batch = batch['polygon']
        else:
            gt_polyons_batch = batch['init_polygon']
        tags_batch = batch['tags']
        pred_polygons_batch = np.array(boxes[0])
        pred_scores_batch = np.array(boxes[1])
        for polygons, pred_polygons, pred_scores, tags in \
                zip(gt_polyons_batch, pred_polygons_batch, pred_scores_batch,
                    tags_batch):
            gt = [
                dict(points=polygons[i], ignore=tags[i])
                for i in range(len(polygons))
            ]
            if self.polygon:
                pred = [
                    dict(points=pred_polygons[i])
                    for i in range(len(pred_polygons))
                ]
            else:
                pred = []
                if isinstance(pred_polygons, list):
                    pred_polygons = np.array(pred_polygons)
                for i in range(pred_polygons.shape[0]):
                    pred.append(dict(points=pred_polygons[i, :, :].tolist()))

            results.append(self.evaluator.evaluate_image(gt, pred))
        return results

    def validate_measure(self,
                         batch,
                         boxes,
                         is_output_polygon=False,
                         training=False):
        return self.measure(batch, boxes, training)

    def evaluate_measure(self, batch, boxes, traing=False):
        return self.measure(batch, boxes, traing), \
               np.linspace(0, batch['image'].shape[0]).tolist()

    def gather_measure(self, raw_metrics):
        raw_metrics = [
            image_metrics for batch_metrics in raw_metrics
            for image_metrics in batch_metrics
        ]

        result = self.evaluator.combine_results(raw_metrics)

        precision = AverageMeter()
        recall = AverageMeter()
        fmeasure = AverageMeter()

        precision.update(result['precision'], n=len(raw_metrics))
        recall.update(result['recall'], n=len(raw_metrics))
        fmeasure_score = 2 * precision.val * recall.val / \
                         (precision.val + recall.val + 1e-8)
        fmeasure.update(fmeasure_score)

        return {'precision': precision, 'recall': recall, 'fmeasure': fmeasure}


@METRICS.register_module
class QuadMeasurer:

    def __init__(self, polygon=False, measure_phase=('train', 'val')):
        self.recalls = []
        self.precisions = []
        self.fmean = []
        self.polygon = polygon
        self.evaluator = DetectionIoUEvaluator()
        self.history = []
        self.phase = measure_phase

    def reset(self):
        self.history = []

    def measure(self, batch, boxes, training=False):
        '''
        batch: (image, polygons, ignore_tags
        batch: a dict produced by dataloaders.
            image: tensor of shape (N, C, H, W).
            polygons: tensor of shape (N, K, 4, 2), the polygons
             of objective regions.
            tags: tensor of shape (N, K), indicates whether a region is
             ignorable or not.
            shape: the original shape of images.
            filename: the original filenames of images.
        output: (polygons, ...)
        '''
        results = []
        if training:
            gt_polyons_batch = batch['polygon']
        else:
            gt_polyons_batch = batch['init_polygon']
        tags_batch = batch['tags']
        pred_polygons_batch = np.array(boxes[0])
        pred_scores_batch = np.array(boxes[1])
        for polygons, pred_polygons, pred_scores, tags in \
                zip(gt_polyons_batch, pred_polygons_batch, pred_scores_batch,
                    tags_batch):
            gt = [
                dict(points=polygons[i], ignore=tags[i])
                for i in range(len(polygons))
            ]
            if self.polygon:
                pred = [
                    dict(points=pred_polygons[i])
                    for i in range(len(pred_polygons))
                ]
            else:
                pred = []
                if isinstance(pred_polygons, list):
                    pred_polygons = np.array(pred_polygons)
                for i in range(pred_polygons.shape[0]):
                    pred.append(dict(points=pred_polygons[i, :, :].tolist()))

            results.append(self.evaluator.evaluate_image(gt, pred))
        self.history.append(results)

    @property
    def metrics(self):

        raw_metrics = [
            image_metrics for batch_metrics in self.history
            for image_metrics in batch_metrics
        ]

        result = self.evaluator.combine_results(raw_metrics)
        precision = AverageMeter()
        recall = AverageMeter()
        fmeasure = AverageMeter()

        precision.update(result['precision'], n=len(raw_metrics))
        recall.update(result['recall'], n=len(raw_metrics))
        fmeasure_score = 2 * precision.val * recall.val / \
                         (precision.val + recall.val + 1e-8)
        fmeasure.update(fmeasure_score)

        return {
            'precision': precision.avg,
            'recall': recall.avg,
            'fmeasure': fmeasure.avg,
        }
