import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import cv2
import torch
import numpy as np

from vedastd.transforms import build_transform
import albumentations as alb

is_show = True


def imshow(name, img):
    if is_show:
        cv2.namedWindow(name, 0)
        cv2.imshow(name, img)
    else:
        pass


def test_cfg(cfg, imgs: dict = None):
    image = cv2.imread(r'D:\DATA_ALL\STD\IC5\ch4_training_images\img_1.jpg')
    print(image.shape)
    imshow('input_image', image)
    polygon = [[377, 117, 463, 117, 465, 130, 378, 130],
               [493, 115, 519, 115, 519, 131, 493, 131],
               [374, 155, 409, 155, 409, 170, 374, 170],
               [492, 151, 551, 151, 551, 170, 492, 170],
               [376, 198, 422, 198, 422, 212, 376, 212],
               [494, 190, 539, 189, 539, 205, 494, 206],
               [374, 1, 494, 0, 492, 85, 372, 86]]

    polygon = [np.array(p).reshape(-1, 2) for p in polygon]
    tags = [True, True, False, True, True, False, False]
    each_len = [0, 4, 4 * 2, 4 * 3, 4 * 4, 4 * 5, 4 * 6, 4 * 7]
    show_polygon(image, polygon, tags, name='init_polygon')
    if imgs is not None:
        assert 'image' in imgs
        image = imgs['image']
        assert 'polygon' in imgs
        polygon = imgs['polygon']
        polygon = [np.array(p).reshape(-1, 2) for p in polygon]
        assert 'tags' in imgs
        tags = imgs['tags']
        assert 'each_len' in imgs
        each_len = imgs['each_len']

    transforms = build_transform(cfg)
    tr_out = transforms(image=image,
                        keypoints=np.array(polygon).reshape(-1, 2),
                        tags=tags,
                        each_len=each_len)

    return tr_out


def show_polygon(image, polygon, tag=None, name=None):
    show_img = image.copy()
    for idx, p in enumerate(polygon):
        if tag is not None:
            color = (0, 255, 0) if tag[idx] else (255, 0, 0)
        else:
            color = (0, 0, 255)

        cv2.polylines(show_img, [p.reshape(-1, 1, 2).astype(np.int)], True, color, 10)
    if name is None:
        name = 'polygon'
    imshow(name, show_img)


def show_trans_results(trans_res, name_lists=None):
    if 'polygon' in trans_res:
        show_polygon(trans_res['image'], trans_res['polygon'], trans_res.get('tags'))
    if name_lists is None:
        imshow('1', trans_res['image'])
        if 'masks' in trans_res:
            for idx, mask in enumerate(trans_res['masks']):
                imshow('%s' % (idx + 2), mask)
    else:
        for idx, name in enumerate(name_lists):
            assert name in trans_res
            current_show = trans_res[name]
            if isinstance(current_show, list):
                for idx2, img in enumerate(current_show):
                    imshow('%s_%s' % (name, idx2), img)
            else:
                if current_show.shape[-1] != 1 and current_show.shape[-1] != 3:
                    for idx3 in range(current_show.shape[-1]):
                        imshow('%s_%s' % (name, idx3), current_show[:, :, idx3])
                else:
                    imshow('%s' % name, current_show)
    if is_show:
        cv2.waitKey()


def test_shrink_map():
    cfg = [
        dict(type='KeypointsToPolygon'),
        dict(type='MakeShrinkMap', ratios=[1.0],
             max_shr=20, min_text_size=8, p=1), ]
    tr_out = test_cfg(cfg)
    show_trans_results(tr_out)


def test_border_map():
    cfg = [
        dict(type='KeypointsToPolygon'),
        dict(type='MakeBorderMap', shrink_ratio=0.8)
    ]
    tr_out = test_cfg(cfg)
    show_trans_results(tr_out)


def test_maskmarker_grouping():
    cfg = [
        dict(type='KeypointsToPolygon'),
        dict(type='MakeShrinkMap', ratios=[1.0],
             max_shr=20, min_text_size=8, p=1),
        dict(type='MaskMarker', name='gt'),
        dict(type='MakeBorderMap', shrink_ratio=0.8),
        dict(type='MaskMarker', name='border'),
        dict(type='Grouping', channel_first=False),
    ]
    tr_out = test_cfg(cfg)
    show_trans_results(tr_out, ['gt', 'border'])


def test_random_crop():
    cfg = [
        # dict(type='RandomCropBasedOnBox', p=1, min_crop_side_ratio=0.001),
        dict(type='Rotate', limit=10, border_mode='constant', value=0),
        dict(type='IAAFliplr', p=0.5),
        dict(type='KeypointsToPolygon'),
    ]
    tr_out = test_cfg(cfg)
    show_trans_results(tr_out)


def test_keypoints_to_polygon():
    cfg = [
        dict(type='KeypointsToPolygon'),
    ]
    tr_out = test_cfg(cfg)
    assert 'polygon' in tr_out


def test_resize_pad():
    cfg = [
        dict(type='RandomScale', scale_range=(0.5, 2.0), interpolation='bilinear', p=1),
        # dict(type='LongestMaxSize', max_size=640, interpolation='bilinear', p=1),
        dict(type='PadorResize', min_height=640, min_width=640, border_mode='constant', value=0),
        dict(type='KeypointsToPolygon'),
    ]
    tr_out = test_cfg(cfg)
    print(tr_out['image'].shape)
    show_trans_results(tr_out)


def test_make_map_other_transform():
    """This function is build to test situation that make all maps first and then do other transforms.
    But it will cause some bad case such as some box should be ignored but not. You'd better do transforms
    like `test_other_transform_make_map`.
    """
    cfg = [
        dict(type='KeypointsToPolygon'),
        dict(type='MakeShrinkMap', ratios=[1.0],
             max_shr=0.6, min_text_size=8, p=1),
        dict(type='MaskMarker', name='gt'),
        dict(type='MakeBorderMap', shrink_ratio=0.8),
        dict(type='MaskMarker', name='border'),
        dict(type='LongestMaxSize', max_size=640, interpolation='bilinear', p=1),
        dict(type='Rotate', interpolation='bilinear', border_mode='constant', value=0, p=1),
        dict(type='PadIfNeeded', min_height=640, min_width=640, border_mode='constant',
             value=0),
        dict(type='KeypointsToPolygon'),
        dict(type='Grouping', channel_first=False),
    ]
    tr_out = test_cfg(cfg)
    show_trans_results(tr_out, ['image', 'gt', 'border'])


def test_other_transform_make_map():
    """Recommend transform order."""

    cfg = [
        # dict(type='LongestMaxSize', max_size=640, interpolation='bilinear', p=1),
        dict(type='RandomScale', scale_range=(0.5, 3.0)),
        dict(type='Rotate', limit=10, border_mode='constant', value=0),
        dict(type='IAAFliplr', p=0.5),
        # dict(type='Rotate', interpolation='bilinear', border_mode='constant', value=0, p=1),
        dict(type='RandomCropBasedOnBox'),
        # dict(type='PadIfNeeded', min_height=640, min_width=640, border_mode='constant',
        #      value=0),
        dict(type='KeypointsToPolygon'),
        dict(type='MakeShrinkMap', ratios=[0.6],
             max_shr=20, min_text_size=8, p=1),
        dict(type='MaskMarker', name='gt'),
        dict(type='MakeBorderMap', shrink_ratio=0.4),
        dict(type='MaskMarker', name='border'),
        dict(type='Grouping', channel_first=False),
    ]
    tr_out = test_cfg(cfg)
    show_trans_results(tr_out, ['image', 'gt', 'border'])


def test_totensor():
    """A complete pipeline. """
    cfg = [
        dict(type='LongestMaxSize', max_size=640, interpolation='bilinear', p=1),
        dict(type='Rotate', interpolation='bilinear', border_mode='constant', value=0, p=1),
        dict(type='PadIfNeeded', min_height=640, min_width=640, border_mode='constant',
             value=0),
        dict(type='KeypointsToPolygon'),
        dict(type='MakeShrinkMap', ratios=[1.0],
             max_shr=20, min_text_size=8, p=1),
        dict(type='MaskMarker', name='gt'),
        dict(type='MakeBorderMap', shrink_ratio=0.8),
        dict(type='MaskMarker', name='border'),
        dict(type='Grouping', channel_first=False),
        dict(type='ToTensor'),
        # dict(type='Grouping', channel_first=True),
    ]
    tr_out = test_cfg(cfg)
    assert isinstance(tr_out['image'], torch.Tensor)
    print('Shape of image, ', tr_out['image'].shape)
    assert isinstance(tr_out['gt'], torch.Tensor)
    print('Shape of name gt, ', tr_out['gt'].shape)
    assert isinstance(tr_out['border'], torch.Tensor)
    print('Shape of name border, ', tr_out['border'].shape)


if __name__ == '__main__':
    for i in range(100):
        # test_random_crop()
        # test_border_map()
        # test_totensor()
        # test_maskmarker_grouping()
        # test_make_map_other_transform()
        # test_resize_pad()
        # test_shrink_map()
        # test_keypoints_to_polygon()
        test_other_transform_make_map()
