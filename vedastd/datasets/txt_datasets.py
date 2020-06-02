import os

import cv2

from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module
class TxtDataset(BaseDataset):

    def __init__(self, img_root, gt_root, transforms):
        super(TxtDataset, self).__init__(img_root, gt_root, transforms)

    def get_needed_item(self):
        need_items = []
        with open(self.img_root, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                poly_list, tag_list = self.load_ann(line)
                need_tuple = (os.path.join(self.img_root, line), poly_list, tag_list)
                need_items.append(need_tuple)

        return need_items

    def load_ann(self, name):
        poly_list = []
        tag_list = []
        with open(os.path.join(self.gt_root, name), 'r') as f:
            for line in f.readlines():
                line = line.split(',')
                line = list(map(int, line))
                poly_list.append(line[:-1])
                tag_list.append(line[-1])

        return poly_list, tag_list

    def pre_transforms(self, result):
        result['img_root'] = self.img_root
        result['gt_root'] = self.gt_root

    def __getitem__(self, index):
        im_path, polys, tags = self.item_lists[index]
        results = dict()
        self.pre_transforms(results)
        image = cv2.imread(im_path)
        results['init_image'] = image
        results['polygon'] = polys
        results['tags'] = tags

        if self.transforms:
            results = self.transforms(results)

        return results
