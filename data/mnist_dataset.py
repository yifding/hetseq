import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.data = None
        self.path = path
        self.read_data(self.path)

    def read_data(self, path):
        self.data = torch.load(path)
        self._len = len(data[0][0])

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        return [self.data[0][index], self.data[1][index]]

    def __len__(self):
        return self._len

    def collater(self, samples):
        # For now only supports datasets with same underlying collater implementations
        # print("samples", type(samples))
        if len(samples) == 0:
            return None
        else:
            return default_collate(samples)
