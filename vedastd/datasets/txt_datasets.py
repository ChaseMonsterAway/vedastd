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
                poly_list, tag_list, each_len = self.load_ann(line)
                need_tuple = (os.path.join(self.img_root, line), poly_list, tag_list, each_len)
                need_items.append(need_tuple)

        return need_items

    def load_ann(self, name):
        poly_list = []
        each_len = [0]
        tag_list = []
        with open(os.path.join(self.gt_root, name + '.txt'), 'r') as f:
            for line in f.readlines():
                line = line.split(',')
                line = list(map(float, line))
                poly = list(zip(line[0::2], line[1::2]))
                poly_list += poly
                each_len.append(len(poly_list))
                if line[-1] == 0:
                    tag_list.append(True)
                else:
                    tag_list.append(False)

        return poly_list, tag_list, each_len

    def pre_transforms(self, result):
        result['img_root'] = self.img_root
        result['gt_root'] = self.gt_root
        result['mask_type'] = []
        result['image_type'] = ['input']

    def __getitem__(self, index):
        im_path, polys, tags, each_len = self.item_lists[index]
        results = dict()
        self.pre_transforms(results)
        image = cv2.imread(im_path)
        shape = image.shape

        if self.transforms:
            results = self.transforms(image=image, keypoints=polys, masks=None, each_len=each_len, tags=tags)
        results['keypoints'] = [results['keypoints'][each_len[i - 1]:each_len[i]]
                                for i in range(1, len(each_len))]
        results['shape'] = np.array(shape[:2])
        results['tags'] = np.array(tags)

        return results
