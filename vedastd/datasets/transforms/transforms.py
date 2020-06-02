import torch
import numpy as np
import cv2
from shapely.geometry import Polygon
import pyclipper
from .registry import TRANSFORMS

CV2_MODE = {
    'bilinear': cv2.INTER_LINEAR,
    'nearest': cv2.INTER_NEAREST,
    'cubic': cv2.INTER_AREA,
}

"""
predefined keys:  模型输入：input
                  标    签：..._label
                  data = dict(input=None,
                              a_label=None,
                              b_label=None,
                              ...) 
"""


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


@TRANSFORMS.register_module
class Normalize:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        return data


@TRANSFORMS.register_module
class ToTensor:
    def __call__(self, data):
        return data


@TRANSFORMS.register_module
class Resize:
    def __init__(self, size, keep_ratio=False, keep_long=False, mode='cubic'):
        self.size = size
        self.keep_ratio = keep_ratio
        self.keep_long = keep_long
        self.mode = mode

    def __call__(self, data):
        return data


@TRANSFORMS.register_module
class PadIfNeeded:
    def __init__(self, size, pad_value=0):
        self.height = size[0]
        self.width = size[1]
        self.pad_value = pad_value

    def __call__(self, data):
        return data


class MakeShrinkMap:

    def __init__(self, ratios: list, max_shr: (int, float)):
        self.ratios = ratios
        self.max_shr = max_shr

    def __call__(self, data: dict):
        assert 'polygon' in data, f'{self} need polygon to generate ' \
                                  f'shrink map'
        shrink_maps = []
        polygons = data['polygon']
        image = data['input']
        h, w = image.shape[:2]
        for ratio in self.ratios:
            ratio = 1 - ratio ** 2
            current_shrink_map = np.zeros(shape=(1, h, w), dtype=np.float32)
            for polygon in polygons:
                polygon = np.array(polygon).reshape(-1, 2)
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
                cv2.fillPoly(current_shrink_map[0], [shrink_box.astype(np.int32)], 1)

            shrink_maps.append(current_shrink_map)
        data.update(shrink_map_label=shrink_maps)

        return data


class MakeBoarderMap:

    def __init__(self, shrink_ratio, thresh_min, thresh_max):
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max

    def __call__(self, data):
        r'''
        required keys:
            image, polygons, ignore_tags
        adding keys:
            thresh_map, thresh_mask
        '''
        image = data['input']
        polygons = data['polygon']
        tags = data['tags']
        canvas = np.zeros(image.shape[:2], dtype=np.float32)
        mask = np.zeros(image.shape[:2], dtype=np.float32)

        for i in range(len(polygons)):
            if not tags[i]:
                continue
            self.draw_border_map(polygons[i], canvas, mask=mask)
        canvas = canvas * (self.thresh_max - self.thresh_min) + self.thresh_min
        data['boarder_label'] = canvas
        data['boarder_mask_label'] = mask

        return data

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
            np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

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
        ex_point_1 = (int(round(point_1[0] + (point_1[0] - point_2[0]) * (1 + self.shrink_ratio))),
                      int(round(point_1[1] + (point_1[1] - point_2[1]) * (1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_1), tuple(point_1),
                 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        ex_point_2 = (int(round(point_2[0] + (point_2[0] - point_1[0]) * (1 + self.shrink_ratio))),
                      int(round(point_2[1] + (point_2[1] - point_1[1]) * (1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_2), tuple(point_2),
                 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        return ex_point_1, ex_point_2


class MakeSegMap:

    def __init__(self, min_text_size, shrink_ratio):
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio

    def __call__(self, data):

        image = data['input']
        polygons = data['polygon']
        tags = data['tags']
        h, w = image.shape[:2]
        polygons, tags = self.validate_polygons(polygons, tags, h, w)

        gt = np.zeros((1, h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(len(polygons)):
            polygon = polygons[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])

            if not tags[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, polygon.astype(
                    np.int32)[np.newaxis, :, :], 0)
                tags[i] = False
            else:
                polygon_shape = Polygon(polygon)
                distance = polygon_shape.area * \
                           (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
                subject = [tuple(l) for l in polygons[i]]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND,
                                pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)
                if shrinked == []:
                    cv2.fillPoly(mask, polygon.astype(
                        np.int32)[np.newaxis, :, :], 0)
                    tags[i] = False
                    continue
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
                cv2.fillPoly(gt[0], [shrinked.astype(np.int32)], 1)

        data.update(input=image,
                    polygon=polygons,
                    gt=gt, mask=mask)
        return data

    def validate_polygons(self, polygons, tags, h, w):
        '''
        polygons (numpy.array, required): of shape (num_instances, num_points, 2)
        '''
        if len(polygons) == 0:
            return polygons, tags
        assert len(polygons) == len(tags)
        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            area = self.polygon_area(np.array(polygons[i]).reshape(-1, 2))
            if abs(area) < 1:
                tags[i] = False
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        return polygons, tags

    def polygon_area(self, polygon):
        edge = 0
        for i in range(polygon.shape[0]):
            next_index = (i + 1) % polygon.shape[0]
            edge += (polygon[next_index, 0] - polygon[i, 0]) * (polygon[next_index, 1] - polygon[i, 1])

        return edge / 2.


@TRANSFORMS.register_module
class FilterKeys:

    def __call__(self, data: dict):
        for key in data.keys():
            if 'input' not in key and 'label' not in key:
                data.pop(key)

        return data
