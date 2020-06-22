# modify from clovaai

from torch.utils.data import DataLoader, ConcatDataset
from .registry import DATALOADERS


@DATALOADERS.register_module
class BaseDataloader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None):
        ndataset = ConcatDataset(dataset)
        super(BaseDataloader, self).__init__(ndataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             sampler=sampler,
                                             batch_sampler=batch_sampler,
                                             num_workers=num_workers,
                                             collate_fn=collate_fn,
                                             pin_memory=pin_memory,
                                             drop_last=drop_last,
                                             timeout=timeout,
                                             worker_init_fn=worker_init_fn,
                                             multiprocessing_context=multiprocessing_context)
