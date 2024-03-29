import numpy as np

from ..utils.icdar15 import DetectionIoUEvaluator


class QuadMeasurer:

    def __init__(self, polygon=False, box_thresh=0.6):
        self.recalls = []
        self.precisions = []
        self.fmean = []
        self.polygon = polygon
        self.box_thresh = box_thresh
        self.evaluator = DetectionIoUEvaluator()

    def reset(self):
        self.recalls = []
        self.precisions = []
        self.fmean = []

    def measure(self, batch, output):
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
        gt_polyons_batch = batch['polygon']
        ignore_tags_batch = batch['tags']
        pred_polygons_batch = np.array(output[0])
        pred_scores_batch = np.array(output[1])
        for polygons, pred_polygons, pred_scores, ignore_tags in \
                zip(gt_polyons_batch, pred_polygons_batch, pred_scores_batch, ignore_tags_batch):
            gt = [
                dict(points=polygons[i], ignore=ignore_tags[i])
                for i in range(len(polygons))
            ]
            if self.polygon:
                pred = [
                    dict(points=pred_polygons[i])
                    for i in range(len(pred_polygons))
                ]
            else:
                pred = []
                for i in range(pred_polygons.shape[0]):
                    assert pred_scores[i] >= self.box_thresh
                    pred.append(dict(points=pred_polygons[i, :, :].tolist()))
            results.append(self.evaluator.evaluate_image(gt, pred))
        return results

    def validate_measure(self,
                         batch,
                         output,
                         is_output_polygon=False,
                         box_thresh=0.6):
        return self.measure(batch, output)

    def evaluate_measure(self, batch, output):
        return self.measure(batch, output), \
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
