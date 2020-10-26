import logging
import threading
from collections import OrderedDict

import albumentations as alb
import cv2
import numpy as np
import torch
import pyclipper
from shapely.geometry import Polygon

from .registry import TRANSFORMS

CV2_MODE = {
    'bilinear': cv2.INTER_LINEAR,
    'nearest': cv2.INTER_NEAREST,
    'cubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
}

CV2_BORDER_MODE = {
    'constant': cv2.BORDER_CONSTANT,
    'reflect': cv2.BORDER_REFLECT,
    'reflect101': cv2.BORDER_REFLECT101,
    'replicate': cv2.BORDER_REPLICATE,
}

logger = logging.getLogger('Transforms')


@TRANSFORMS.register_module
class MaskMarker(alb.NoOp):
    _mask_index = [0]
    _names = []
    _instance_lock = threading.Lock()
    _hooker_count = -1

    def __init__(self, name):
        super(MaskMarker, self).__init__(always_apply=True, p=1)
        self.name = name

    def __call__(self, force_apply=False, **kwargs):
        len_masks = len(kwargs.get('masks', []))
        if len_masks > self._mask_index[-1]:
            self._mask_index.append(len_masks)

        return kwargs

    def __new__(cls, *args, **kwargs):
        if not hasattr(MaskMarker, "_instance"):
            with MaskMarker._instance_lock:
                if not hasattr(MaskMarker, "_instance"):
                    MaskMarker._instance = super().__new__(cls)
                    MaskMarker._hooker_count += 1
        if kwargs.get('name'):
            assert kwargs.get('name') not in MaskMarker._names
            MaskMarker._names.append(kwargs.get('name'))
        else:
            MaskMarker._names.append(cls._hooker_count)

        return MaskMarker._instance

    @classmethod
    def get_names(cls):
        return cls._names

    @classmethod
    def get_index(cls):
        return cls._mask_index


@TRANSFORMS.register_module
class MakeShrinkMap(alb.NoOp):
    """Make mask map and gt map based on the given ratios.

    Examples:
        >>> masks = MakeShrinkMap(ratios=[1], max_shr=0.1, min_text_size=4, p=1)()
        >>> # generate gt mask and corresponding effective mask.
        >>> # In specificly, gt_mask = masks[:masks.shape[0]//2], effective_mask = masks[masks.shape[0]//2:]
    """

    def __init__(self, ratios: list, max_shr: (int, float), min_text_size: int,
                 always_apply=False, p=0.5):
        super(MakeShrinkMap, self).__init__(always_apply=always_apply, p=p)
        self.ratios = ratios
        self.max_shr = max_shr
        self.min_text_size = min_text_size

    def __call__(self, force_apply=False, **kwargs):
        keypoints = kwargs.get('keypoints')
        each_len = kwargs.get('each_len')
        poly = [np.array(keypoints[each_len[i - 1]:each_len[i]])[:, :2] for i in range(1, len(each_len))]
        shrink_maps = []
        mask_maps = []
        polygons = poly
        image = kwargs.get('image')
        tags = kwargs.get('tags')
        h, w = image.shape[:2]
        for ratio in self.ratios:
            ratio = 1 - ratio ** 2
            current_shrink_map = np.zeros(shape=(h, w, 1), dtype=np.float32)
            current_mask_map = np.ones(shape=(h, w, 1), dtype=np.float32)
            for idx, polygon in enumerate(polygons):
                polygon = np.array(polygon).reshape(-1, 2)
                height = max(polygon[:, 1]) - min(polygon[:, 1])
                width = max(polygon[:, 0]) - min(polygon[:, 0])
                if not tags[idx] or min(height, width) < self.min_text_size:
                    tags[idx] = False
                    cv2.fillPoly(current_mask_map[:, :, 0], [polygon.astype(np.int32)], 0)
                    continue
                p_polygon = Polygon(polygon)
                area = p_polygon.area
                perimeter = p_polygon.length
                pco = pyclipper.PyclipperOffset()
                pco.AddPath(polygon, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                offset = min(area * ratio / perimeter + 0.5, self.max_shr)
                shrink_box = pco.Execute(-offset)
                if not shrink_box or len(shrink_box[0]) <= 2:
                    shrink_box = polygon

                shrink_box = np.array(shrink_box[0]).reshape(-1, 2)
                cv2.fillPoly(current_shrink_map[:, :, 0], [shrink_box.astype(np.int32)], 1)

            shrink_maps.append(current_shrink_map)
            mask_maps.append(current_mask_map)

        kwargs['masks'] = shrink_maps + mask_maps

        return kwargs


@TRANSFORMS.register_module
class RandomCropBasedOnBox(alb.RandomCropNearBBox):
    """Random crop the image based on the bounding boxes."""

    def __init__(self, always_apply=False, p=1.0):
        self.max_tries = 50
        self.min_crop_side_ratio = 0.2

        super(RandomCropBasedOnBox, self).__init__(always_apply=always_apply, p=p)

    @property
    def targets_as_params(self):
        return ['image', 'keypoints', 'tags', 'each_len']

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        keypoints = params['keypoints']
        each_len = params['each_len']
        poly = [np.array(keypoints[each_len[i - 1]:each_len[i]])[:, :2]
                for i in range(1, len(each_len))]
        tags = params['tags']
        all_care_polys = []
        for idx, line in enumerate(poly):
            if tags[idx]:
                all_care_polys.append(line)

        crop_x, crop_y, crop_w, crop_h = self.crop_area(img, all_care_polys)

        return {"x_min": crop_x, "x_max": crop_x + crop_w, "y_min": crop_y, "y_max": crop_y + crop_h}

    @staticmethod
    def is_poly_in_rect(poly, x, y, w, h):
        poly = np.array(poly)
        if poly[:, 0].min() < x or poly[:, 0].max() > x + w:
            return False
        if poly[:, 1].min() < y or poly[:, 1].max() > y + h:
            return False
        return True

    @staticmethod
    def is_poly_outside_rect(poly, x, y, w, h):
        poly = np.array(poly)
        if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
            return True
        if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
            return True
        return False

    @staticmethod
    def split_regions(axis):
        regions = []
        min_axis = 0
        for i in range(1, axis.shape[0]):
            if axis[i] != axis[i - 1] + 1:
                region = axis[min_axis:i]
                min_axis = i
                regions.append(region)
        return regions

    @staticmethod
    def random_select(axis, max_size):
        xx = np.random.choice(axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        xmin = np.clip(xmin, 0, max_size - 1)
        xmax = np.clip(xmax, 0, max_size - 1)
        return xmin, xmax

    @staticmethod
    def region_wise_random_select(regions, max_size):
        selected_index = list(np.random.choice(len(regions), 2))
        selected_values = []
        for index in selected_index:
            axis = regions[index]
            xx = int(np.random.choice(axis, size=1))
            selected_values.append(xx)
        xmin = min(selected_values)
        xmax = max(selected_values)
        return xmin, xmax

    def crop_area(self, img, polys):
        h, w, _ = img.shape
        h_array = np.zeros(h, dtype=np.int32)
        w_array = np.zeros(w, dtype=np.int32)
        for points in polys:
            points = np.round(points, decimals=0).astype(np.int32)
            minx = np.min(points[:, 0])
            maxx = np.max(points[:, 0])
            w_array[minx:maxx] = 1
            miny = np.min(points[:, 1])
            maxy = np.max(points[:, 1])
            h_array[miny:maxy] = 1
        # ensure the cropped area not across a text
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]

        if len(h_axis) == 0 or len(w_axis) == 0:
            return 0, 0, w, h

        h_regions = self.split_regions(h_axis)
        w_regions = self.split_regions(w_axis)

        for i in range(self.max_tries):
            if len(w_regions) > 1:
                xmin, xmax = self.region_wise_random_select(w_regions, w)
            else:
                xmin, xmax = self.random_select(w_axis, w)
            if len(h_regions) > 1:
                ymin, ymax = self.region_wise_random_select(h_regions, h)
            else:
                ymin, ymax = self.random_select(h_axis, h)

            if xmax - xmin < self.min_crop_side_ratio * w or ymax - ymin < self.min_crop_side_ratio * h:
                # area too small
                continue
            num_poly_in_rect = 0
            for poly in polys:
                if not self.is_poly_outside_rect(poly, xmin, ymin, xmax - xmin, ymax - ymin):
                    num_poly_in_rect += 1
                    break

            if num_poly_in_rect > 0:
                return xmin, ymin, xmax - xmin, ymax - ymin

        return 0, 0, w, h


@TRANSFORMS.register_module
class ToTensor(alb.DualTransform):
    """Transfer data from numpy.ndarray to torch.Tensor."""

    def __init__(self, always_apply=True, p=1):
        super(ToTensor, self).__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        # img np.ndarray
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img

    def apply_to_mask(self, img, **params):
        img = torch.from_numpy(img)
        return img

    def apply_to_keypoint(self, keypoint, **params):
        return torch.from_numpy(np.array(keypoint))

    def apply_to_keypoints(self, keypoints, **params):
        return torch.stack([self.apply_to_keypoint(keypoint) for keypoint in keypoints])


@TRANSFORMS.register_module
class FilterKeys(alb.NoOp):
    """Remove keys."""

    def __init__(self, op_names):
        super(FilterKeys, self).__init__(always_apply=True, p=1)
        self.op_names = op_names if isinstance(op_names, (list, tuple)) else [op_names]

    def __call__(self, force_apply=False, **kwargs):
        for op in self.op_names:
            if op in kwargs:
                kwargs.pop(op)
            else:
                logger.info(f"{op} isn't existed.")

        return kwargs


@TRANSFORMS.register_module
class Grouping(alb.NoOp):
    """Grouping is used to divide masks in different groups based on MaskMarker."""

    def __init__(self):
        super(Grouping, self).__init__(always_apply=True, p=1)

    def __call__(self, force_apply=False, **kwargs):

        assert 'masks' in kwargs
        groups = {}
        assert len(MaskMarker.get_names()) == len(MaskMarker.get_index()) - 1
        for idx, (name, midx) in enumerate(zip(MaskMarker.get_names(), MaskMarker.get_index()[1:])):
            groups[name] = kwargs.get('masks')[MaskMarker.get_index()[idx]:midx]
        kwargs.update(**groups)
        return kwargs
