import cv2
import torch
import numpy as np

from vedastd.datasets.transforms.transforms import MakeShrinkMap, MakeBoarderMap, FilterKeys, Normalize, ToTensor, \
    Resize, RandomRotation, RandomFlip


def tensor_to_img(t_img: torch.Tensor):
    img = t_img.cpu().numpy()

    return img


def main():
    image = cv2.imread(r'D:\DATA_ALL\STD\IC5\ch4_test_images\img_1.jpg')
    polygon = [[790, 302, 903, 304, 902, 335, 790, 335],
               [822, 288, 872, 286, 871, 298, 823, 300],
               [669, 139, 693, 140, 693, 154, 669, 153],
               [700, 141, 723, 142, 723, 155, 701, 154],
               [668, 157, 693, 158, 693, 170, 668, 170],
               [636, 155, 661, 156, 662, 169, 636, 168]]
    polygon = [np.array(p).reshape(-1, 2) for p in polygon]
    tags = [True, True, False, True, False, True]
    dummy_data = dict(input=image,
                      polygon=polygon,
                      tags=tags,
                      image_type=['input'],
                      mask_type=[],
                      )

    msm = MakeShrinkMap(ratios=[0.4], max_shr=20, min_text_size=8, prefix='seg')
    msm2 = MakeShrinkMap(ratios=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4], max_shr=20, min_text_size=8, prefix='shrink')
    mbm = MakeBoarderMap(shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7)
    filt = FilterKeys()
    totensor = ToTensor()
    norm = Normalize(key='input')
    rtion = RandomRotation(angles=(-10, 10), p=1)
    rs = Resize(keep_ratio=True, size=(600, 400), img_mode='nearest', mask_mode='nearest')
    rf = RandomFlip(p=1, horizaontal=False, vertical=False)

    data2 = msm(dummy_data)
    data2 = msm2(data2)
    data2 = mbm(data2)
    data2 = rs(data2)
    data2 = rtion(data2)
    data2 = rf(data2)
    data2 = filt(data2)
    data2 = totensor(data2)
    data2 = norm(data2)
    # data2 = mmsm(dummy_data)
    # data3 = mbm(dummy_data)
    # for idx, shrink in enumerate(data['shrink_map_label']):
    #     print(np.sum(shrink))
    #     cv2.imshow('img_%s'%idx, shrink[0].astype(np.uint8) * 255)
    # cv2.waitKey()

    cv2.imshow('thm', (255 * tensor_to_img(data2['boarder_map_label'])).astype(np.uint8))
    cv2.imshow('thmsk', (255 * tensor_to_img(data2['boarder_mask_label'])).astype(np.uint8))
    cv2.imshow('mask', (255 * tensor_to_img(data2['seg_mask_label'][0][0])).astype(np.uint8))
    cv2.imshow('gt', (255 * tensor_to_img(data2['seg_map_label'][0][0])).astype(np.uint8))
    cv2.imshow('shrink_mask', (255 * tensor_to_img(data2['shrink_mask_label'][0][0])).astype(np.uint8))
    cv2.imshow('shrink_gt', (255 * tensor_to_img(data2['shrink_map_label'][0][0])).astype(np.uint8))
    cv2.waitKey()

    print('done')


if __name__ == '__main__':
    main()
