import cv2
import numpy as np

from vedastd.datasets.transforms.transforms import MakeShrinkMap, MakeSegMap, MakeBoarderMap


def main():
    image = cv2.imread(r'D:\DATA_ALL\STD\IC5\ch4_test_images\img_1.jpg')
    polygon = [[790, 302, 903, 304, 902, 335, 790, 335],
               [822, 288, 872, 286, 871, 298, 823, 300],
               [669, 139, 693, 140, 693, 154, 669, 153],
               [700, 141, 723, 142, 723, 155, 701, 154],
               [668, 157, 693, 158, 693, 170, 668, 170],
               [636, 155, 661, 156, 662, 169, 636, 168]]
    polygon = [np.array(p).reshape(-1, 2) for p in polygon]
    tags = [True] * 6
    dummy_data = dict(input=image,
                      polygon=polygon,
                      tags=tags,
                      )
    msm = MakeShrinkMap(ratios=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4], max_shr=20)
    mmsm = MakeSegMap(min_text_size=2, shrink_ratio=0.4)
    mbm = MakeBoarderMap(shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7)

    data = msm(dummy_data)
    data2 = mmsm(dummy_data)
    data3 = mbm(dummy_data)
    for idx, shrink in enumerate(data['shrink_map_label']):
        print(np.sum(shrink))
        cv2.imshow('img_%s'%idx, shrink[0].astype(np.uint8) * 255)
        # cv2.waitKey()

    cv2.imshow('thm', (255*data['thresh_map_label']).astype(np.uint8))
    cv2.imshow('thmsk', (255 * data['thresh_mask_label']).astype(np.uint8))
    cv2.imshow('mask', (255 * data['mask']).astype(np.uint8))
    cv2.waitKey()

    print('done')


if __name__ == '__main__':
    main()
