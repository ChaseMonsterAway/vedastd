import os

import cv2
import numpy as np

from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module
class TxtDataset(BaseDataset):

    def __init__(self, img_root, gt_root, txt_file, transforms):
        self.txt_file = txt_file
        super(TxtDataset, self).__init__(img_root, gt_root, transforms)

    def get_needed_item(self):
        need_items = []
        with open(self.txt_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                poly_list, tag_list = self.load_ann(line)
                need_tuple = (os.path.join(self.img_root, line), poly_list, tag_list)
                need_items.append(need_tuple)

        return need_items

    def load_ann(self, name):
        poly_list = []
        tag_list = []
        with open(os.path.join(self.gt_root, name + '.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\ufeff').split(',')
                line[:-1] = map(int, line[:-1])
                poly = np.array(line[:-1]).reshape(-1, 2)
                poly_list.append(poly)
                if line[-1][0] == '#':
                    tag_list.append(False)
                else:
                    tag_list.append(True)
                """
                line = list(map(int, line))
                poly = np.array(line[:-1]).reshape(-1, 2)
                poly_list.append(poly)
                if line[-1] == 0:
                    tag_list.append(True)
                else:
                    tag_list.append(False)
                """
        return poly_list, tag_list

    def pre_transforms(self, result):
        result['img_root'] = self.img_root
        result['gt_root'] = self.gt_root
        result['mask_type'] = []
        result['image_type'] = ['input']

    def __getitem__(self, index):
        im_path, polys, tags = self.item_lists[index]
        results = dict()
        self.pre_transforms(results)
        image = cv2.imread(im_path)
        shape = image.shape
        results['input'] = image
        results['polygon'] = np.array(polys)
        results['shape'] = np.array(shape[:2])
        results['tags'] = np.array(tags)

        if self.transforms:
            results = self.transforms(results)

        return results
