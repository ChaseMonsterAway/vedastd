import cv2
import numpy as np
import pdb
import pyclipper
import torch
from shapely.geometry import Polygon

from .registry import POSTPROCESS
from .utils import pse


@POSTPROCESS.register_module
class Postprocessor:

    def __init__(self,
                 thresh=0.3,
                 box_thresh=0.7,
                 max_candidates=100,
                 unclip_ratio=1.5,
                 name='binary_map',
                 min_size=3,
                 debug=False):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.dest = name
        self.ur = unclip_ratio
        self.min_size = min_size
        self.debug = debug

    def set_params(self, **kwargs):
        self.__init__(**kwargs)

    def __call__(self, batch, _pred, training=False):
        images: torch.Tensor = batch['image']
        pred: torch.Tensor = _pred[self.dest]
        segmentation: torch.Tensor = self.binarize(pred)
        boxes_batch = []
        scores_batch = []
        for batch_index in range(images.size(0)):
            init_shape: np.ndarray = batch['shape'][batch_index]
            height, width = init_shape

            resize_shape: np.ndarray = batch['resized_shape'][batch_index]
            # if not training:
            #     pdb.set_trace()
            resized_h, resized_w = resize_shape
            hscale = resized_h / height
            wscale = resized_w / width
            if training:
                height, width = resized_h, resized_w
                wscale, hscale = 1, 1
            # refer to threshold
            boxes, scores = self.boxes_from_bitmap(
                _pred[self.dest][batch_index], segmentation[batch_index],
                wscale, hscale, height, width)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch

    def binarize(self, pred):
        # cv2.imshow('pred', pred[0,0].cpu().numpy())
        return pred > self.thresh

    def boxes_from_bitmap(self, pred, _bitmap, wscale, hscale, h, w):
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
        _, contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8),
                                          cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        if self.debug:
            bitmap = cv2.cvtColor(pred * 255, cv2.COLOR_GRAY2BGR)

        # TODO, SELECT TOP SELF.MAX_CANDIDATES DIRECTLY
        for contour in contours[:self.max_candidates]:
            points, sside = self.get_mini_boxes(contour)

            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))

            if self.debug:
                points = points.astype(np.int32)
                bitmap = cv2.polylines(bitmap, [points.reshape(-1, 2)], True,
                                       (255, 0, 0), 3)
                bitmap = cv2.putText(bitmap, str(round(score, 3)),
                                     (points[:, 0].min(), points[:, 1].min()),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                     (0, 255, 0))
            # if self.debug:
            #     cv2.imshow('mask', bitmap.astype(np.uint8))
            #     cv2.waitKey()
            # cv2.destroyAllWindows()
            # print(score)
            if self.box_thresh > score:
                continue
            scores.append(np.array(score))
            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / wscale), 0, w)
            box[:, 1] = np.clip(np.round(box[:, 1] / hscale), 0, h)
            boxes.append(np.array(box.tolist()))

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

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
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


@POSTPROCESS.register_module
class PsePostprocessor:

    def __init__(self,
                 thresh=1.0,
                 min_kernel_area=5,
                 min_score=0.93,
                 max_candidates=10,
                 min_area=100,
                 resize=False,
                 name=('pred_text_map', 'pred_kernels_map'),
                 debug=False):
        self.binary_th = thresh
        self.max_candidates = max_candidates
        self.min_area = min_area
        self.resize = resize
        self.dest = name
        self.min_kernel_area = min_kernel_area
        self.min_score = min_score
        self.debug = debug

    def __call__(self, batch, _pred):
        images = batch['image']
        outputs = torch.cat((_pred[self.dest[0]], _pred[self.dest[1]]), dim=1)

        score = torch.sigmoid(outputs[:, 0, :, :])
        cv2.imshow('1', score[0].cpu().numpy())
        cv2.waitKey()
        outputs = (torch.sign(outputs - self.binary_th) + 1) / 2

        text = outputs[:, 0, :, :]
        kernels = outputs * text[:, None, :, :]

        score = score.data.cpu().numpy().astype(np.float32)
        # text = text.data.cpu().numpy().astype(np.uint8)
        kernels = kernels.data.cpu().numpy().astype(np.uint8)

        boxes_batch = []
        scores_batch = []
        for batch_index in range(images.size(0)):
            init_shape = batch['shape'][batch_index]
            if isinstance(init_shape, torch.Tensor):
                height, width = init_shape.data.numpy()
            else:
                height, width = init_shape
            resize_shape = batch['resized_shape']
            if isinstance(resize_shape, torch.Tensor):
                resized_h, resized_w = resize_shape[batch_index].data.numpy()
            else:
                # pdb.set_trace()
                resized_h, resized_w = resize_shape[batch_index]
            hscale = resized_h.data / height
            wscale = resized_w.data / width
            # scale = max(height, width) / max(resized_h.data, resized_w.data)
            # scale = batch['ratio'][batch_index].data.numpy()
            # if self.debug:
            # show_img = images[batch_index].permute(1, 2, 0).numpy()
            # show_img = (show_img - np.min(show_img)) / (np.max(show_img) - np.min(show_img))
            # show_img = (show_img * 255).astype(np.uint8)
            # show_img *= 58
            # show_img += 110
            # cv2.imshow('input', show_img)
            boxes, scores = self.boxes_from_bitmap(kernels[batch_index],
                                                   score[batch_index], wscale,
                                                   hscale, height, width)
            # for box in boxes:
            #     cv2.rectangle(show_img, tuple(box[0]), tuple(box[2]), (0, 255, 0))
            # cv2.imshow('ii', show_img)
            # cv2.waitKey()

            boxes_batch.append(np.array(boxes))
            scores_batch.append(np.array(scores))
        return boxes_batch, scores_batch

    def boxes_from_bitmap(self, kernels, score, wscale, hscale, h, w):
        '''
        kernels: map with shape (7, H, W),
            whose values are binarized as {0, 1}
        '''
        assert kernels.shape[0] == 7
        pred = pse(kernels, self.min_kernel_area)
        label = pred
        label_num = np.max(label) + 1
        b_list = []
        s_list = []
        for i in range(1, label_num):
            points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]

            if points.shape[0] < self.min_area:
                continue

            s_i = np.mean(score[label == i])
            if s_i < self.min_score:
                continue

            rect = cv2.minAreaRect(points)
            bbox_i = cv2.boxPoints(rect)
            bbox_i[:, 0] = bbox_i[:, 0] / wscale
            bbox_i[:, 1] = bbox_i[:, 1] / hscale
            bbox_i = bbox_i.astype('int32')
            bbox_i[:, 0] = np.clip(np.round(bbox_i[:, 0]), 0, w)
            bbox_i[:, 1] = np.clip(np.round(bbox_i[:, 1]), 0, h)
            b_list.append(bbox_i.tolist())
            s_list.append(s_i)
        return b_list, s_list
