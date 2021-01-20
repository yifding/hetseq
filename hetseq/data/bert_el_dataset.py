from functools import lru_cache

import torch
import numpy as np


class BertELDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, args):
        self.args = args
        self.dataset = dataset

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return len(self.dataset)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return np.arange(len(self.dataset))

    def num_tokens(self, index: int):
        return len(self.dataset[index]['labels'])

    def collater(self, samples):
        if len(samples) == 0:
            return None
        else:
            return self.args.data_collator(samples)

    def set_epoch(self, epoch):
        pass


