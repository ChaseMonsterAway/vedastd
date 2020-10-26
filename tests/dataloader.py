import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import cv2
import numpy as np

from vedastd.transforms import build_transform
from vedastd.datasets import build_datasets
from vedastd.dataloaders import build_dataloader
from vedastd.dataloaders.collate_fn import build_collate_fn

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
    each_len = [0, 4, 4 * 2, 4 * 3, 4 * 4, 4 * 5]
    tr = [
        dict(type='MakeShrinkMap', ratios=[1], max_shr=0.4, min_text_size=4, p=1),
        dict(type='MaskMarker', name='gt'),
        dict(type='MakeShrinkMap', ratios=[0.9, 0.8, 0.7], max_shr=0.4, min_text_size=4, p=1),
        dict(type='MaskMarker', name='shrink'),
        dict(type='LongestMaxSize', max_size=640, interpolation='bilinear', p=1),
        dict(type='PadIfNeeded', min_height=512, min_width=640, border_mode='constant',
             value=0),
    ]
    transforms = build_transform(tr)

    dt = [dict(type='TxtDataset',
               img_root=r'D:\DATA_ALL\STD\IC5\ch4_training_images',
               gt_root=r'D:\DATA_ALL\STD\IC5\ch4_training_localization_transcription_gt',
               txt_file=r'D:\DATA_ALL\STD\IC5\train.txt',
               ignore_tag='###',
               )]

    # tr_out = transforms(image=image, keypoints=np.array(polygon).reshape(-1, 2), tags=tags, each_len=each_len)
    collate_fn = dict(type='BaseCollate', stack_keys=['image', 'gt', 'shrink'])
    clf = build_collate_fn(collate_fn)
    dataset = build_datasets(dt, dict(transforms=transforms))

    dl = dict(type='BaseDataloader', batch_size=2)
    dataloader = build_dataloader(dl, dict(dataset=dataset, collate_fn=clf))
    for batch in dataloader:
        for idx in range(batch['image'].shape[0]):
            image = batch['image'][idx].data.numpy()
            # image = (image - np.min(image)) / (np.max(image) - np.min(image))
            image = image.astype(np.uint8)
            polygons = batch['polygon'][idx]
            for polygon in polygons:
                cv2.rectangle(image, (int(polygon[0, 0]), int(polygon[0, 1])),
                              (int(polygon[2, 0]), int(polygon[2, 1])), (0, 255, 0), 2)
            cv2.imshow('i', image)
            cv2.waitKey()
        print(batch.keys())

    print('done')
