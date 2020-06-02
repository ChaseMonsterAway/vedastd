import numpy as np
from torch.utils.data import DataLoader


class MyDatasets(object):

    def __init__(self):
        pass

    def __getitem__(self, item):
        result = {}
        result['image'] = np.zeros((3, 224, 224), dtype=np.float)
        result['kernel'] = np.zeros((7, 224, 224), dtype=np.float)
        result['mask'] = [np.zeros((224, 224), dtype=np.float), np.zeros((224, 224), dtype=np.float)]

        return result

    def __len__(self):
        return 100


if __name__ == '__main__':
    datasets = MyDatasets()
    dataloader = DataLoader(dataset=datasets,
                            batch_size=2,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True)
    for batch in dataloader:
        print('done')
