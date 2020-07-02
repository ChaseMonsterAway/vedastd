import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

from .registry import POSTPROCESS


@POSTPROCESS.register_module
class Postprocessor:
    def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=100, unclip_ratio=1.5,
                 resize=False, name='binary_map', min_size=3, debug=False):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.resize = resize
        self.dest = name
        self.ur = unclip_ratio
        self.min_size = min_size
        self.debug = debug

    def __call__(self, batch, _pred):
        images = batch['input']
        ratio = batch['ratio'].item()
        pred = _pred[self.dest]
        segmentation = self.binarize(pred)
        boxes_batch = []
        scores_batch = []
        for batch_index in range(images.size(0)):
            height, width = batch['shape'][batch_index].data.numpy()
            boxes, scores = self.boxes_from_bitmap(
                _pred['binary_map'][batch_index],
                segmentation[batch_index], ratio, height, width)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch

    def binarize(self, pred):
        return pred > self.thresh

    def boxes_from_bitmap(self, pred, _bitmap, ratio, h, w):
        '''
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        '''
        assert _bitmap.size(0) == 1
        bitmap = _bitmap.data.cpu().numpy()[0]  # The first channel
        pred = pred.cpu().detach().numpy()[0]
        height, width = bitmap.shape
        boxes = []
        scores = []
        _, contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        if self.debug:
            bitmap = cv2.cvtColor(pred * 255, cv2.COLOR_GRAY2BGR)

        # TO DO
        for contour in contours[:self.max_candidates]:
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))

            if self.debug:
                points = points.astype(np.int32)
                bitmap = cv2.polylines(
                    bitmap, [points.reshape(-1, 2)], True, (255, 0, 0), 3)
                bitmap = cv2.putText(
                    bitmap, str(round(score, 3)),
                    (points[:, 0].min(), points[:, 1].min()),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            if self.box_thresh > score:
                continue
            scores.append(score)
            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            if not self.resize:
                ratio = 1

            box[:, 0] = np.clip(
                np.round(box[:, 0] / ratio), 0, w)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / ratio), 0, h)
            boxes.append(box.tolist())

        if self.debug:
            cv2.imshow('mask', bitmap.astype(np.uint8))
            cv2.waitKey()

        return boxes, scores

    def unclip(self, box):
        poly = Polygon(box)
        distance = poly.area * self.ur / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2],
               points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    @staticmethod
    def box_score(bitmap, box):
        '''
        naive version of box score computation,
        only for helping principle understand.
        '''
        mask = np.zeros_like(bitmap, dtype=np.uint8)
        cv2.fillPoly(mask, box.reshape(1, 4, 2).astype(np.int32), 1)
        return cv2.mean(bitmap, mask)[0]

    @staticmethod
    def box_score_fast(bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
