import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import cv2 # noqa 402

from vedastd.transforms import build_transform, MaskMarker # noqa 402
from vedastd.datasets import build_datasets # noqa 402
from tests.utils import tensor_to_numpy, imshow, show_polygon # noqa 402

is_show = True


def generate_transform():
    tr = [
        dict(type='LongestMaxSize', max_size=640, interpolation='bilinear', p=1),
        dict(type='Rotate', interpolation='bilinear', border_mode='constant', value=0, p=1),
        dict(type='PadIfNeeded', min_height=640, min_width=640, border_mode='constant',
             value=0),
        dict(type='KeypointsToPolygon'),
        dict(type='MakeBorderMap', shrink_ratio=0.8),
        dict(type='MaskMarker', name='border'),
        dict(type='MakeShrinkMap', ratios=[1.0],
             max_shr=20, min_text_size=8, p=1),
        dict(type='MaskMarker', name='gt'),
        # dict(type='Grouping', channel_first=False),
        dict(type='ToTensor'),
        dict(type='Grouping', channel_first=True),
    ]
    transforms = build_transform(tr)

    return transforms


def test_cfg(cfg):
    transforms = generate_transform()
    dataset = build_datasets(cfg, dict(transforms=transforms))
    mask_names = MaskMarker.get_names()

    return dataset, mask_names


def test_TxtDataset():
    dt = [dict(type='TxtDataset',
               img_root=r'D:\DATA_ALL\STD\IC5\ch4_training_images',
               gt_root=r'D:\DATA_ALL\STD\IC5\ch4_training_localization_transcription_gt',
               txt_file=r'D:\DATA_ALL\STD\IC5\train.txt',
               ignore_tag='###',
               )]
    datasets, mask_names = test_cfg(dt)
    assert isinstance(datasets, list)
    for idx, dataset in enumerate(datasets):
        print('--' * 20, idx, 'start', '--' * 20)
        for data in dataset:
            if is_show:
                # show image
                for name in mask_names:
                    if name not in data:
                        continue
                    mask = data[name]
                    for idx2 in range(mask.size(0)):
                        print(idx2, mask[idx2, :, :].size())
                        imshow('%s_%s' % (name, idx2), tensor_to_numpy(mask[idx2, :, :]), is_show)
                img = data['image']
                imshow('image', tensor_to_numpy(img), is_show)
                print('img shape,', tensor_to_numpy(img).shape)
                assert 'polygon' in data
                print(data['polygon'])
                show_polygon(tensor_to_numpy(img), data['polygon'], data.get('tags'), name='polygon', is_show=is_show)
                cv2.waitKey()
            else:
                continue
                # print(data)
        print('--' * 20, idx, 'end！', '--' * 20)
    print('done')


if __name__ == '__main__':
    for i in range(100):
        test_TxtDataset()
