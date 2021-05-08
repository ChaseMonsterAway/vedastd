import cv2
import numpy as np
import queue


def pse(kernals, min_area):
    kernal_num = len(kernals)
    pred = np.zeros(kernals[0].shape, dtype='int32')

    label_num, label = cv2.connectedComponents(
        kernals[kernal_num - 1], connectivity=4)

    for label_idx in range(1, label_num):
        if np.sum(label == label_idx) < min_area:
            label[label == label_idx] = 0

    first_queue = queue.Queue(maxsize=0)
    next_queue = queue.Queue(maxsize=0)
    points = np.array(np.where(label > 0)).transpose((1, 0))

    for point_idx in range(points.shape[0]):
        x, y = points[point_idx, 0], points[point_idx, 1]
        l = label[x, y]
        first_queue.put((x, y, l))
        pred[x, y] = l

    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    for kernal_idx in range(kernal_num - 2, -1, -1):
        kernal = kernals[kernal_idx].copy()
        while not first_queue.empty():
            (x, y, l) = first_queue.get()

            is_edge = True
            for j in range(4):
                tmpx = x + dx[j]
                tmpy = y + dy[j]
                if tmpx < 0 or tmpx >= kernal.shape[
                        0] or tmpy < 0 or tmpy >= kernal.shape[1]:
                    continue
                if kernal[tmpx, tmpy] == 0 or pred[tmpx, tmpy] > 0:
                    continue

                first_queue.put((tmpx, tmpy, l))
                pred[tmpx, tmpy] = l
                is_edge = False
            if is_edge:
                next_queue.put((x, y, l))

        first_queue, next_queue = next_queue, first_queue

    return pred


if __name__ == '__main__':
    import torch

    outputs = torch.rand(1, 7, 224, 224)
    score = torch.sigmoid(outputs[:, 0, :, :])
    outputs = (torch.sign(outputs - 1) + 1) / 2

    text = outputs[:, 0, :, :]
    kernels = outputs[:, 0:7, :, :] * text

    score = score.data.cpu().numpy()[0].astype(np.float32)
    text = text.data.cpu().numpy()[0].astype(np.uint8)
    kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)

    min_area = 5
    print(pse(kernels, min_area))
