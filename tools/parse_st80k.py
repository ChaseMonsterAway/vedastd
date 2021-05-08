import argparse
import os

import numpy as np
import scipy.io as sio
from tqdm import tqdm


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mat', help='Mat file of st.')
    parser.add_argument('-out_txt', help='Directory for output txt files.')
    parser.add_argument(
        '-label_txt', help='Directory to save txt file of each instances.')
    args = parser.parse_args()
    return args


def main():
    args = parse()
    gt_mat = sio.loadmat(args.mat)
    boxes = gt_mat['wordBB'][0]
    label_txts = gt_mat['txt'][0]
    img_names = gt_mat['imnames'][0]
    ttt = []
    with open(args.out_txt, 'r') as f:
        for idx, name in enumerate(tqdm(img_names)):
            label = label_txts[idx]
            texts = []
            for l in label:
                l = l.strip().split('\n')
                for l1 in l:
                    if ' ' in l1:
                        texts += l1.split(' ')
                    else:
                        texts.append(l1)
            texts = [t for t in texts if len(t) != 0]
            ttt += texts
            box = boxes[idx].astype(np.float32)
            if box.ndim == 2:
                box = box[:, :, np.newaxis]
            box = box.transpose(2, 1, 0)

            assert box.shape[0] == len(texts), f'The length of box and text' \
                                               f'should be same. Current ' \
                                               f'unequal index is {idx}, ' \
                                               f'img name is {img_names}'

            f.writelines(name[0])
            f.writelines('\n')
            f.flush()
            directory = name[0].split('/')[0]
            if not os.path.exists(os.path.join(args.label_txt, directory)):
                os.makedirs(os.path.join(args.label_txt, directory))
            with open(os.path.join(args.label_txt, name[0] + '.txt'),
                      'w') as f2:
                for idx1 in range(box.shape[0]):
                    loc = box[idx1].reshape(-1).tolist()
                    text = texts[idx1]
                    f2.writelines(f'{",".join(list(map(str, loc)))},{text}\n')


if __name__ == '__main__':
    main()
