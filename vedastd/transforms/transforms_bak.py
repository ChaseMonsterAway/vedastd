import albumentations as alb
import cv2
import numpy as np
import pyclipper
import random
import torch
from functools import partial
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
alb.LongestMaxSize
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
class FilterKeys:

    def __init__(self, need_keys: list):
        self.need_keys = need_keys

    def __call__(self, data: dict):
        keys = list(data.keys())

        for key in keys:
            if key not in self.need_keys:
                data.pop(key)

        return data


@TRANSFORMS.register_module
class MakeShrinkMap:

    def __init__(self, ratios: list, max_shr: (int, float), min_text_size: int,
                 prefix: str):
        self.ratios = ratios
        self.max_shr = max_shr
        self.min_text_size = min_text_size
        self.prefix = prefix

    def __call__(self, data: dict):
        assert 'polygon' in data, f'{self} need polygon to generate ' \
                                  f'shrink map'
        shrink_maps = []
        mask_maps = []
        polygons = data['polygon']
        image = data['input']
        tags = data['tags']
        h, w = image.shape[:2]
        for ratio in self.ratios:
            ratio = 1 - ratio**2
            current_shrink_map = np.zeros(shape=(h, w, 1), dtype=np.float32)
            current_mask_map = np.ones(shape=(h, w, 1), dtype=np.float32)
            for idx, polygon in enumerate(polygons):
                polygon = np.array(polygon).reshape(-1, 2)
                height = max(polygon[:, 1]) - min(polygon[:, 1])
                width = max(polygon[:, 0]) - min(polygon[:, 0])
                if not tags[idx] or min(height, width) < self.min_text_size:
                    tags[idx] = False
                    cv2.fillPoly(current_mask_map[:, :, 0],
                                 [polygon.astype(np.int32)], 0)
                    continue
                p_polygon = Polygon(polygon)
                area = p_polygon.area
                perimeter = p_polygon.length
                pco = pyclipper.PyclipperOffset()
                pco.AddPath(polygon, pyclipper.JT_ROUND,
                            pyclipper.ET_CLOSEDPOLYGON)
                offset = min(area * ratio / perimeter + 0.5, self.max_shr)
                shrink_box = pco.Execute(-offset)
                if not shrink_box or len(shrink_box[0]) <= 2:
                    shrink_box = polygon

                shrink_box = np.array(shrink_box[0]).reshape(-1, 2)
                cv2.fillPoly(current_shrink_map[:, :, 0],
                             [shrink_box.astype(np.int32)], 1)
            shrink_maps.append(current_shrink_map)
            mask_maps.append(current_mask_map)
        # TO DO, not list, but np.ndarray
        data[self.prefix + '_map'] = shrink_maps
        data[self.prefix + '_mask'] = mask_maps
        data['mask_type'].append(self.prefix + '_map')
        data['mask_type'].append(self.prefix + '_mask')
        # print(data)

        #for id, item in enumerate(data['text_map']):
        #    cv2.imshow('map', item)
        #    cv2.waitKey()
        # print(data[self.prefix + '_map'][0].shape)
        #cv2.imshow('map', data[self.prefix + '_map'][0])
        #cv2.waitKey()
        #print(data[self.prefix + '_map'][0].shape)
        #cv2.imshow('mask', data[self.prefix + '_mask'][0])
        #cv2.waitKey()
        return data


@TRANSFORMS.register_module
class MakeBoarderMap:

    def __init__(self, shrink_ratio, thresh_min=0.3, thresh_max=0.7):
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max

    def __call__(self, data):

        image = data['input']
        polygons = data['polygon']
        tags = data['tags']
        h, w = image.shape[:2]
        canvas = np.zeros((h, w, 1), dtype=np.float32)
        mask = np.zeros((h, w, 1), dtype=np.float32)

        for i in range(len(polygons)):
            if not tags[i]:
                continue
            self.draw_border_map(
                polygons[i], canvas[:, :, 0], mask=mask[:, :, 0])
            canvas = canvas * (self.thresh_max -
                               self.thresh_min) + self.thresh_min
            data['boarder_map'] = canvas
            data['boarder_mask'] = mask

            data['mask_type'].append('boarder_map')
            data['mask_type'].append('boarder_mask')

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
            np.linspace(0, width - 1, num=width).reshape(1, width),
            (height, width))
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1),
            (height, width))

        distance_map = np.zeros((polygon.shape[0], height, width),
                                dtype=np.float32)
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
            1 - distance_map[ymin_valid - ymin:ymax_valid - ymax + height,
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
        square_distance_1 = np.square(xs - point_1[0]) + np.square(ys -
                                                                   point_1[1])
        square_distance_2 = np.square(xs - point_2[0]) + np.square(ys -
                                                                   point_2[1])
        square_distance = np.square(point_1[0] -
                                    point_2[0]) + np.square(point_1[1] -
                                                            point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) / \
                (2 * np.sqrt(square_distance_1 * square_distance_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 * square_sin /
                         square_distance)

        result[cosin < 0] = np.sqrt(
            np.fmin(square_distance_1, square_distance_2))[cosin < 0]

        return result

    def extend_line(self, point_1, point_2, result):
        ex_point_1 = (int(
            round(point_1[0] + (point_1[0] - point_2[0]) *
                  (1 + self.shrink_ratio))),
                      int(
                          round(point_1[1] + (point_1[1] - point_2[1]) *
                                (1 + self.shrink_ratio))))
        cv2.line(
            result,
            tuple(ex_point_1),
            tuple(point_1),
            4096.0,
            1,
            lineType=cv2.LINE_AA,
            shift=0)
        ex_point_2 = (int(
            round(point_2[0] + (point_2[0] - point_1[0]) *
                  (1 + self.shrink_ratio))),
                      int(
                          round(point_2[1] + (point_2[1] - point_1[1]) *
                                (1 + self.shrink_ratio))))
        cv2.line(
            result,
            tuple(ex_point_2),
            tuple(point_2),
            4096.0,
            1,
            lineType=cv2.LINE_AA,
            shift=0)
        return ex_point_1, ex_point_2


@TRANSFORMS.register_module
class Normalize:

    def __init__(self,
                 mean=(127.5, 127.5, 127.5),
                 std=(127.5, 127.5, 127.5),
                 key: str = 'input'):
        self.mean = mean
        self.std = std
        self.key = key

    def __call__(self, data):
        assert self.key in data, f'{self.key} is not in data, pls check it'
        image = data[self.key]
        mean = torch.as_tensor(
            self.mean, dtype=torch.float32,
            device=image.device).view(-1, 1, 1)
        std = torch.as_tensor(
            self.std, dtype=torch.float32, device=image.device).view(-1, 1, 1)
        image.sub_(mean).div_(std)

        data[self.key] = image

        return data


@TRANSFORMS.register_module
class PadIfNeeded:

    def __init__(self,
                 factor=32,
                 pad_value=0,
                 img_border_mode='constant',
                 mask_border_mode='constant'):
        self.factor = factor
        self.pad_value = pad_value
        self.img_border_mode = CV2_BORDER_MODE[img_border_mode]
        self.mask_border_mode = CV2_BORDER_MODE[mask_border_mode]

    def _pad(self, img, mode, pad_h, pad_w, vaule=0):
        return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, mode, value=vaule)

    def __call__(self, data):
        img = data['input']
        h, w = img.shape[:-1]

        pad_h = 0 if h % 32 == 0 else 32 - h % 32
        pad_w = 0 if w % 32 == 0 else 32 - w % 32

        mask_type_lists = data['mask_type']
        image_type_lists = data['image_type']
        for key, values in data.items():
            if key in mask_type_lists:
                mode = self.mask_border_mode
            elif key in image_type_lists:
                mode = self.img_border_mode
            else:
                continue
            if isinstance(values, list):
                temp_list = []
                for value in values:
                    new_img = self._pad(value, mode, pad_h, pad_w, 0)
                    if len(new_img.shape) == 2:
                        new_img = new_img[:, :, np.newaxis]
                    temp_list.append(new_img)
                data[key] = temp_list
            else:
                new_img = self._pad(values, mode, pad_h, pad_w, 0)
                if len(new_img.shape) == 2:
                    new_img = new_img[:, :, np.newaxis]
                data[key] = new_img
        data['ratio'] = 1.0
        return data


@TRANSFORMS.register_module
class RandomFlip:

    def __init__(self, p, horizontal, vertical):
        self.p = p
        self.h = horizontal
        self.v = vertical

    @staticmethod
    def _random_horizontal_flip(img, flag=False):
        if flag:
            img = np.flip(img, axis=1).copy()
        return img

    @staticmethod
    def _random_vertical_flip(img, flag=False):
        if flag:
            img = np.flip(img, axis=0).copy()
        return img

    def _flip(self, image, hflag, vflag):
        image = self._random_horizontal_flip(image, hflag)
        image = self._random_vertical_flip(image, vflag)

        return image

    def __call__(self, data: dict):
        hflag = True if random.random() < self.p else False
        vflag = True if random.random() < self.p else False
        hflag = hflag & self.h
        vflag = vflag & self.v

        flip = partial(self._flip, hflag=hflag, vflag=vflag)
        mask_type_lists = data['mask_type']
        image_type_lists = data['image_type']
        for key, values in data.items():
            if key in mask_type_lists or key in image_type_lists:
                if isinstance(values, list):
                    temp_list = []
                    for value in values:
                        value = flip(image=value)
                        temp_list.append(value)
                    data[key] = temp_list
                else:
                    values = flip(image=values)
                    data[key] = values

        return data


@TRANSFORMS.register_module
class RandomRotation(object):

    def __init__(self,
                 img_value=0,
                 mask_value=0,
                 angles: tuple = None,
                 p=0.5,
                 img_mode='bilinear',
                 img_border_mode='constant',
                 mask_mode='bilinear',
                 mask_border_mode='constant'):
        self.p = 1 - p
        self.angles = angles if angles is not None else (0, 360)
        self.img_value = img_value
        self.mask_value = mask_value
        self.img_border_mode = CV2_BORDER_MODE[img_border_mode]
        self.img_mode = CV2_MODE[img_mode]
        self.mask_border_mode = CV2_BORDER_MODE[mask_border_mode]
        self.mask_mode = CV2_MODE[mask_mode]

    def affine(self, image, rotation_mat, h, w, mode, border_mode,
               border_value):
        ndims = image.ndim
        new_img = cv2.warpAffine(
            image,
            rotation_mat, (w, h),
            flags=mode,
            borderMode=border_mode,
            borderValue=border_value)
        if new_img.ndim != ndims:
            new_img = new_img[:, :, np.newaxis]
        return new_img

    def __call__(self, data):
        if random.random() < self.p:
            return data
        degree = random.randint(self.angles[0], self.angles[1])
        h, w = data['input'].shape[:2]
        mask_type_lists = data['mask_type']
        image_type_lists = data['image_type']
        rotation_mat = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 1)
        affine = partial(self.affine, rotation_mat=rotation_mat, w=w, h=h)

        for key, values in data.items():
            if key in mask_type_lists:
                mode = self.mask_mode
                border_mode = self.mask_border_mode
                pad_value = self.mask_value
            elif key in image_type_lists:
                mode = self.img_mode
                border_mode = self.img_border_mode
                pad_value = self.img_value
            else:
                continue

            if isinstance(values, list):
                temp_list = []
                for value in values:
                    new_img = affine(
                        image=value,
                        mode=mode,
                        border_mode=border_mode,
                        border_value=pad_value)
                    temp_list.append(new_img)
                data[key] = temp_list
            else:
                new_img = affine(
                    image=values,
                    mode=mode,
                    border_mode=border_mode,
                    border_value=pad_value)
                data[key] = new_img
        # for item in data['kernels_mask']:
        #     cv2.imshow('map', item)
        #     cv2.waitKey()
        #     print(item.shape)
        return data


@TRANSFORMS.register_module
class Resize:

    def __init__(self,
                 size,
                 keep_ratio=False,
                 random_scale=False,
                 img_mode='cubic',
                 mask_mode='nearest',
                 max_size=1280,
                 scale_list=(0.5, 1.0, 2.0, 3.0)):
        self.h = size[0]
        self.w = size[1]
        self.keep_ratio = keep_ratio
        self.random_scale = random_scale
        self.img_mode = img_mode
        self.mask_mode = mask_mode
        self.max_size = max_size
        self.scale_list = scale_list
        self.random_value = None

    def reset_random(self):
        self.random_value = np.random.choice(self.scale_list)

    def _get_target_size(self, image):
        h, w = image.shape[:2]
        if self.keep_ratio:
            ratio = min(self.h / h, self.w / w)
            target_size = int(h * ratio), int(w * ratio)
        else:
            if self.random_scale:
                if max(h, w) > self.max_size:
                    scale = self.max_size / max(h, w)
                    h, w = int(h * scale), int(w * scale)

                # scale = np.random.choice(self.scale_list)
                scale = self.random_value
                if min(h, w) * scale <= self.h:
                    scale = (self.h + 10) * 1.0 / min(h, w)
                target_size = int(h * scale), int(w * scale)
            else:
                target_size = self.h, self.w
            ratio = None
        return target_size, ratio

    def _resize(self, image, mode):
        ndims = image.ndim
        target_size, ratio = self._get_target_size(image)
        new_image = cv2.resize(
            image, target_size[::-1], interpolation=CV2_MODE[mode])

        if new_image.ndim != ndims:
            new_image = new_image[:, :, np.newaxis]

        return new_image, ratio

    def __call__(self, data):
        mask_type_lists = data['mask_type']
        image_type_lists = data['image_type']
        self.reset_random()
        for key, values in data.items():
            if key in mask_type_lists:
                mode = self.mask_mode
            elif key in image_type_lists:
                mode = self.img_mode
            else:
                continue
            if isinstance(values, list):
                temp_list = []
                for value in values:
                    new_img, ratio = self._resize(value, mode)
                    temp_list.append(new_img)
                data[key] = temp_list
            else:
                new_img, ratio = self._resize(values, mode)
                data[key] = new_img
        data['ratio'] = ratio
        # cv2.imshow('map', data['text_map'][0])
        # cv2.waitKey()
        # print(data['text_map'][0].shape)
        # cv2.imshow('mask', data['text_mask'][0])
        # cv2.waitKey()
        # print(data['text_mask'][0].shape)
        # for item in data['kernels_map']:
        #    cv2.imshow('mask', item)
        #    cv2.waitKey()
        #    print(item.shape)
        return data


@TRANSFORMS.register_module
class RandomCrop:

    def __init__(self, size, p=3.0 / 8.0, prefix='text'):
        self.h = size[0]
        self.w = size[1]
        self.p = p
        self.prefix = prefix

    def get_crop_size(self, text_map):
        h, w = text_map.shape[0:2]
        if random.random() > self.p and np.max(text_map) > 0:
            tl = np.min(np.where(text_map > 0), axis=1) - (self.h, self.w, 0)
            tl[tl < 0] = 0
            br = np.max(np.where(text_map > 0), axis=1) - (self.h, self.w, 0)
            br[br < 0] = 0
            br[0] = min(br[0], h - self.h)
            br[1] = min(br[1], w - self.w)

            i = random.randint(tl[0], br[0])
            j = random.randint(tl[1], br[1])
        else:
            i = random.randint(0, h - self.h)
            j = random.randint(0, w - self.w)
        return i, j

    def __call__(self, data):
        mask_type_lists = data['mask_type']
        image_type_lists = data['image_type']
        text_map = data[self.prefix + '_map'][0]
        if text_map.shape[0] == self.h and text_map.shape[1] == self.w:
            return data

        i, j = self.get_crop_size(text_map)
        for key, values in data.items():
            if key in mask_type_lists or key in image_type_lists:
                if isinstance(values, list):
                    temp_list = []
                    for value in values:
                        new_img = value[i:i + self.h, j:j + self.w, :]
                        temp_list.append(new_img)
                    data[key] = temp_list
                else:
                    new_img = values[i:i + self.h, j:j + self.w]
                    data[key] = new_img
        # for item in data['kernels_map']:
        #     cv2.imshow('mask', item)
        #     cv2.waitKey()
        #     print(item.shape)
        return data


"""@TRANSFORMS.register_module
class Resize:
    def __init__(self, size, keep_ratio=False, img_mode='cubic', mask_mode='nearest'):
        self.h = size[0]
        self.w = size[1]
        self.keep_ratio = keep_ratio
        self.img_mode = img_mode
        self.mask_mode = mask_mode

    def get_target_size(self, image):
        h, w = image.shape[:2]
        if self.keep_ratio:
            ratio = min(self.h / h, self.w / w)
            target_size = int(h * ratio), int(w * ratio)
        else:
            target_size = self.h, self.w
            ratio = None
        return target_size, ratio

    def _resize(self, image, mode):
        ndims = image.ndim
        target_size, ratio = self.get_target_size(image)
        new_image = cv2.resize(image, target_size[::-1], interpolation=CV2_MODE[mode])

        if new_image.ndim != ndims:
            new_image = new_image[:, :, np.newaxis]

        return new_image, ratio

    def __call__(self, data):
        mask_type_lists = data['mask_type']
        image_type_lists = data['image_type']
        for key, values in data.items():
            if key in mask_type_lists:
                mode = self.mask_mode
            elif key in image_type_lists:
                mode = self.img_mode
            else:
                continue
            if isinstance(values, list):
                temp_list = []
                for value in values:
                    new_img, ratio = self._resize(value, mode)
                    temp_list.append(new_img)
                data[key] = temp_list
            else:
                new_img, ratio = self._resize(values, mode)
                data[key] = new_img
        data['ratio'] = ratio

        return data"""


@TRANSFORMS.register_module
class KeepLongResize(Resize):

    def __init__(self, *args, **kwargs):
        super(KeepLongResize, self).__init__(*args, **kwargs)
        assert self.keep_ratio is True, 'args: keep_ratio should be True'

    def get_target_size(self, image):
        h, w = image.shape[:2]
        long_edge, short_edge = max(self.h, self.w), min(self.h, self.w)
        ratio = min(long_edge / max(h, w), short_edge / min(h, w))
        target_size = int(h * ratio), int(w * ratio)

        return target_size, ratio


@TRANSFORMS.register_module
class ToTensor:

    def __init__(self, keys: list):
        self.keys = keys

    @staticmethod
    def to_tensor(value):
        if value.ndim == 3:
            if value.shape[0] == 1 or value.shape[0] == 3:
                return torch.from_numpy(value).float()
            else:
                return torch.from_numpy(value).permute(2, 0, 1).float()

        return torch.from_numpy(value).float()

    def __call__(self, data):
        for key, values in data.items():
            if key not in self.keys:
                continue
            if isinstance(values, list):
                temp_list = []
                for value in values:
                    temp_list.append(self.to_tensor(value))
                data[key] = temp_list
            else:
                data[key] = self.to_tensor(values)
        return data


@TRANSFORMS.register_module
class Canvas:

    def __init__(self, size, img_v=255, mask_v=255):
        self.h = size[0]
        self.w = size[1]
        self.img_v = img_v
        self.mask_v = mask_v

    def _canvas(self, image, v):
        ndims = image.ndim
        if ndims == 3:
            h, w, c = image.shape
            new_canvas = np.zeros((self.h, self.w, c))
            new_canvas.fill(v)
            new_canvas[:h, :w, :] = image
        else:
            h, w = image.shape
            new_canvas = np.zeros((self.h, self.w))
            new_canvas.fill(v)
            new_canvas[:h, :w] = image

        return new_canvas

    def __call__(self, data):
        mask_type_lists = data['mask_type']
        image_type_lists = data['image_type']
        for key, values in data.items():
            if key in mask_type_lists:
                v = self.mask_v
            elif key in image_type_lists:
                v = self.img_v
            else:
                continue
            if isinstance(values, list):
                temp_list = []
                for value in values:
                    new_img = self._canvas(value, v)
                    temp_list.append(new_img)
                data[key] = temp_list
            else:
                new_img = self._canvas(values, v)
                data[key] = new_img

        return data
