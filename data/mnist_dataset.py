from functools import lru_cache

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
        self._len = len(self.data[0])
        self.image = self.data[0].unsqueeze(1).float()
        self.label = self.data[1].long()


        # **YD**
        # print(self.data[0].shape, self.data[1].shape)
        # raise ValueError('debugging for data shape')

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        # print(self.image.shape, self.data[1].shape)
        return [self.image[index, :, :, :], self.label[index]]

    def __len__(self):
        return self._len

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return np.arange(len(self))

    def num_tokens(self, index: int):
        return 1

    def collater(self, samples):
        # For now only supports datasets with same underlying collater implementations
        # print("samples", type(samples))
        if len(samples) == 0:
            return None
        else:
            return default_collate(samples)

    def set_epoch(self, epoch):
        pass


if __name__ == '__main__':
    path = '/scratch365/yding4/mnist/MNIST/processed/training.pt'
    dataset = MNISTDataset(path)
    data = torch.load(path)
    print(len(dataset))
    print(data[0].shape, data[1].shape)