import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import cv2
import numpy as np

from vedastd.transforms import build_transform

if __name__ == '__main__':
    image = cv2.imread(r'D:\DATA_ALL\STD\IC5\ch4_test_images\img_1.jpg')
    polygon = [[790, 302, 903, 304, 902, 335, 790, 335],
               [822, 288, 872, 286, 871, 298, 823, 300],
               [669, 139, 693, 140, 693, 154, 669, 153],
               [700, 141, 723, 142, 723, 155, 701, 154],
               [668, 157, 693, 158, 693, 170, 668, 170],
               [636, 155, 661, 156, 662, 169, 636, 168]]
    polygon = [np.array(p).reshape(-1, 2) for p in polygon]
    tags = [True, True, False, True, False, True]
    each_len = [0, 4, 4*2, 4*3, 4*4, 4*5]
    # dummy_data = dict(input=image,
    #                   polygon=polygon,
    #                   tags=tags,
    #                   image_type=['input'],
    #                   mask_t /ype=[],
    #                   )

    tr = [
        dict(type='MakeShrinkMap', ratios=[1], max_shr=0.4, min_text_size=4, p=1),
        dict(type='MaskMarker', name='gt'),
        dict(type='MakeShrinkMap', ratios=[0.9, 0.8, 0.7], max_shr=0.4, min_text_size=4, p=1),
        dict(type='MaskMarker', name='shrink'),
        dict(type='LongestMaxSize', max_size=640, interpolation='bilinear', p=1),
        dict(type='FilterKeys', op_names=['tags', 'each_len']),
        dict(type='ToTensor'),
        dict(type='Grouping'),
    ]
    transforms = build_transform(tr)
    tr_out = transforms(image=image, keypoints=np.array(polygon).reshape(-1, 2), tags=tags, each_len=each_len)
    cv2.imshow('1', tr_out['image'])
    for idx, mask in enumerate(tr_out['masks']):
        cv2.imshow('%s'%(idx+2), mask)
    cv2.waitKey()

    cv2.waitKey()


    print('done')
