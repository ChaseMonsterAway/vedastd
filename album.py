import os
import cv2
import albumentations
import numpy as np


def augment_and_show(aug, image):
    image = aug(image=image)['image']
    cv2.imshow('im', image)
    cv2.waitKey()


def parse_txt(file_names):
    with open(file_names) as f:
        lines = []
        for line in f.readlines():
            line = list(map(int, line.strip().split(',')))
            lines.append(line)

    return lines


def main():
    root = r'D:\DB\express-data\images'
    name = '001_4304989519209_20200403_140644412.jpg'
    image = cv2.imread(os.path.join(root, name))
    lines = parse_txt(os.path.join(r'D:\DB\express-data\train_gts', name + '.txt'))
    polys = [line[:-1] for line in lines]
    aug1 = albumentations.HorizontalFlip(p=1)
    aug2 = albumentations.Rotate(limit=360, border_mode=cv2.BORDER_CONSTANT, value=0, p=1)

    new_loc = []
    for i in range(0, len(polys[0]), 2):
        new_loc.append((polys[0][i], polys[0][i + 1]))
    ceshi = dict(image=image, keypoints=new_loc)
    out = albumentations.Compose([aug1, aug2], p=1,
                                 keypoint_params=albumentations.KeypointParams(format='xy'))(**ceshi)
    out = aug1(image=image, keypoints=new_loc)
    # params = aug1.get_params()
    # params = {'col': 486, 'rows': 917}
    # image = aug1.apply(image, **params)
    # new_loc = aug1.apply_to_keypoint(keypoint=new_loc, **params)

    img = image
    cv2.polylines(img, [np.array(new_loc).reshape(-1, 1, 2)], True, (0, 255, 0))
    cv2.imshow('im', img)
    cv2.waitKey()
    # out = aug(image=image, keypoints=polys[0])
    # augment_and_show(aug, image)
    print('done')


if __name__ == '__main__':
    main()
