import logging
import threading

import albumentations as alb
import cv2
import numpy as np
import pyclipper
import torch
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
    """To assign name and record the masks made."""
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
            if kwargs.get('name') not in MaskMarker._names:
                MaskMarker._names.append(kwargs.get('name'))
            else:
                logger.info(f"{kwargs.get('name')} has already existed. Please use another name.")
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
        # TODO, MAKE IT MORE ELEGANTLY. SO DO MAKEBORDERMAP.
        # TODO, CONSIDERING FUSION CLASS MAKESHRINKMAP & CLASS MAKEBORDERMAP.
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
        if kwargs.get('masks') and kwargs['masks']:
            kwargs['masks'] = kwargs['masks'] + shrink_maps + mask_maps
        else:
            kwargs['masks'] = shrink_maps + mask_maps

        return kwargs


@TRANSFORMS.register_module
class RandomCropBasedOnBox(alb.RandomCropNearBBox):
    """Random crop the image based on the bounding boxes."""

    def __init__(self, always_apply=False, p=1.0):
        self.max_tries = 50
        self.min_crop_side_ratio = 0.2

        super(RandomCropBasedOnBox, self).__init__(always_apply=always_apply, p=p)
        self._crop_area = []

    def __call__(self, force_apply=False, **kwargs):
        kwargs = super(RandomCropBasedOnBox, self).__call__(force_apply, **kwargs)
        keypoints = kwargs['keypoints']
        each_len = kwargs['each_len']
        poly = [np.array(keypoints[each_len[i - 1]:each_len[i]])[:, :2]
                for i in range(1, len(each_len))]
        tags = kwargs['tags']
        for idx, p in enumerate(poly):
            if tags[idx]:
                if self.is_poly_outside_rect(p, *self._crop_area):
                    kwargs['tags'][idx] = False
        return kwargs

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
        self._crop_area = [crop_x, crop_y, crop_w, crop_h]

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
        img = torch.from_numpy(img).permute(2, 0, 1)

        return img

    def apply_to_mask(self, img, **params):
        if img.ndim == 2:
            img = torch.from_numpy(img).unsqueeze(0)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1)

        return img

    def apply_to_keypoints(self, keypoints, **params):
        return keypoints

    def __call__(self, force_apply=False, **kwargs):
        kwargs = super(ToTensor, self).__call__(force_apply, **kwargs)
        if MaskMarker.get_names():
            for name in MaskMarker.get_names():
                if name in kwargs:
                    kwargs[name] = self.apply_to_mask(kwargs[name])
        return kwargs


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

    def __init__(self, channel_first=True):
        super(Grouping, self).__init__(always_apply=True, p=1)
        self.channel_first = channel_first

    def __call__(self, force_apply=False, **kwargs):
        assert 'masks' in kwargs
        groups = {}
        assert len(MaskMarker.get_names()) == len(MaskMarker.get_index()) - 1
        for idx, (name, midx) in enumerate(zip(MaskMarker.get_names(), MaskMarker.get_index()[1:])):
            axis = 0 if self.channel_first else -1
            if isinstance(kwargs['image'], torch.Tensor):
                groups[name] = torch.cat(kwargs['masks'][MaskMarker.get_index()[idx]:midx], axis=axis)
            else:
                groups[name] = np.concatenate(kwargs['masks'][MaskMarker.get_index()[idx]:midx], axis=axis)

        kwargs.update(**groups)
        return kwargs


@TRANSFORMS.register_module
class PadIfNeeded(alb.PadIfNeeded):
    def __init__(self, min_height=1024,
                 min_width=1024,
                 border_mode=cv2.BORDER_REFLECT_101,
                 value=None,
                 mask_value=None,
                 always_apply=False,
                 p=1.0, ):
        border_mode = CV2_BORDER_MODE[border_mode]

        super(PadIfNeeded, self).__init__(min_height=min_height,
                                          min_width=min_width,
                                          border_mode=border_mode,
                                          value=value,
                                          mask_value=mask_value,
                                          always_apply=always_apply,
                                          p=p,
                                          )

    def update_params(self, params, **kwargs):
        params = super(PadIfNeeded, self).update_params(params, **kwargs)
        rows = params["rows"]
        cols = params["cols"]

        if rows < self.min_height:
            h_pad_top = 0
            h_pad_bottom = self.min_height - rows - h_pad_top
        else:
            h_pad_top = 0
            h_pad_bottom = 0

        if cols < self.min_width:
            w_pad_left = 0
            w_pad_right = self.min_width - cols - w_pad_left
        else:
            w_pad_left = 0
            w_pad_right = 0

        params.update(
            {"pad_top": h_pad_top, "pad_bottom": h_pad_bottom, "pad_left": w_pad_left, "pad_right": w_pad_right}
        )
        return params

    def __call__(self, force_apply=False, **kwargs):
        h, w = kwargs['image'].shape[:2]
        kwargs = super().__call__(force_apply, **kwargs)
        kwargs['resized_shape'] = np.array([h, w])
        if 'polygon' in kwargs:
            poly = kwargs['polygon']
            tags = kwargs['tags']
            for idx, p in enumerate(poly):
                if tags[idx]:
                    if np.any(p[:, 0] < 0) or np.any(p[:, 0] > w) or np.any(p[:, 1] < 0) or np.any(p[:, 1] > h):
                        kwargs['tags'][idx] = False

        return kwargs


@TRANSFORMS.register_module
class KeypointsToPolygon(alb.NoOp):
    """This class is used to update the polygon based on keypoints. If you do any other
    transforms after KeypointsToPolygon, you can recall KeypointsToPolygon to generate a
    new polygon."""

    def __init__(self, *args, **kwargs):
        super(KeypointsToPolygon, self).__init__(p=1)

    def __call__(self, force_apply=False, **kwargs):
        assert 'each_len' in kwargs
        each_len = kwargs['each_len']
        polygon = [np.array(kwargs['keypoints'][each_len[i - 1]:each_len[i]])[:, :2].reshape(-1, 2)
                   for i in range(1, len(each_len))]
        kwargs['polygon'] = polygon

        return kwargs


@TRANSFORMS.register_module
class MakeBorderMap(alb.NoOp):
    """Make border map for [db](https://arxiv.org/pdf/1911.08947.pdf). """

    def __init__(self, shrink_ratio, thresh_min=0.3, thresh_max=0.7):
        super(MakeBorderMap, self).__init__(p=1)
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max

    def __call__(self, force_apply=False, **kwargs):
        # TODO, USE OTHER METHODS TO REPRODUCE POLYGON LOCATIONS. CURRENTLY, IT LOOKS AWFUL.

        image = kwargs['image']
        polygons = kwargs['polygon']
        tags = kwargs['tags']
        h, w = image.shape[:2]
        canvas = np.zeros((h, w, 1), dtype=np.float32)
        mask = np.zeros((h, w, 1), dtype=np.float32)

        for i in range(len(polygons)):
            if not tags[i]:
                continue
            try:
                self.draw_border_map(polygons[i], canvas[:, :, 0],
                                     mask=mask[:, :, 0])
            except:
                assert np.any(polygons[i][:, 1] <= 0) or np.any(polygons[i][:, 0] <= 0)
                tags[i] = False
                continue
        canvas = canvas * (
                self.thresh_max - self.thresh_min) + self.thresh_min
        if kwargs.get('masks') and kwargs.get('masks') is not None:
            kwargs['masks'] = kwargs['masks'] + [canvas, mask]
        else:
            kwargs['masks'] = [canvas, mask]

        return kwargs

    def draw_border_map(self, polygon, canvas, mask):
        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2

        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * \
                   (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = np.array(padding.Execute(distance)[0])
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(
            np.linspace(0, width - 1, num=width).reshape(1, width),
            (height, width))
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1),
            (height, width))

        distance_map = np.zeros(
            (polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self.distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[
                ymin_valid - ymin:ymax_valid - ymax + height,
                xmin_valid - xmin:xmax_valid - xmax + width],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

    def distance(self, xs, ys, point_1, point_2):
        '''
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        '''
        height, width = xs.shape[:2]
        square_distance_1 = np.square(
            xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(
            xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(
            point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) / \
                (2 * np.sqrt(square_distance_1 * square_distance_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 *
                         square_sin / square_distance)

        result[cosin < 0] = np.sqrt(np.fmin(
            square_distance_1, square_distance_2))[cosin < 0]

        return result

    def extend_line(self, point_1, point_2, result):
        ex_point_1 = (int(round(
            point_1[0] + (point_1[0] - point_2[0]) * (1 + self.shrink_ratio))),
                      int(round(point_1[1] + (point_1[1] - point_2[1]) * (
                              1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_1), tuple(point_1),
                 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        ex_point_2 = (int(round(
            point_2[0] + (point_2[0] - point_1[0]) * (1 + self.shrink_ratio))),
                      int(round(point_2[1] + (point_2[1] - point_1[1]) * (
                              1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_2), tuple(point_2),
                 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        return ex_point_1, ex_point_2
