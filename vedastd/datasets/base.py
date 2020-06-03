from abc import abstractmethod
import logging

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, img_root, gt_root, transforms):
        self.img_root = img_root
        self.gt_root = gt_root
        self.transforms = transforms
        self.item_lists = self.get_needed_item()
        self.logger = logging.getLogger()
        self.logger.info(f'current dataset length is {len(self)} in {self.img_root}')

    @abstractmethod
    def get_needed_item(self):
        """
        Returns: item_lists
        """
        pass

    def __len__(self):
        return len(self.item_lists)
