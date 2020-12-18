import os
import bisect
from functools import lru_cache

import h5py

import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate


class BertH5pyData(torch.utils.data.Dataset):       # # don't know whether support multiprocess loading?
    def __init__(self, path, max_pred_length=512):
        super(BertH5pyData, self).__init__()
        self.keys = ('input_ids', 'input_mask', 'segment_ids',
                     'masked_lm_positions', 'masked_lm_ids', 'next_sentence_labels')
        self.max_pred_length = max_pred_length
        self.data_file = None
        self.path = path
        self.read_data(path)

    def read_data(self, path):
        with h5py.File(path, "r", libver='latest', swmr=True) as data_file:
            self._len = len(data_file[self.keys[0]])

    def check_index(self, i):
        if i < 0 or i >= self._len:
            raise IndexError('index out of range')

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        with h5py.File(self.path, "r", libver='latest', swmr=True) as data_file:
            self.check_index(index)

            inputs = [data_file[key][index] for key in self.keys]

            [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
                torch.from_numpy(input.astype(np.int64)) if indice < 5 else torch.from_numpy(
                    np.asarray(input.astype(np.int64))) for indice, input in enumerate(inputs)]

            masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
            index = self.max_pred_length
            # store number of  masked tokens in index
            padded_mask_indices = (masked_lm_positions == 0).nonzero()
            if len(padded_mask_indices) != 0:
                index = padded_mask_indices[0].item()
            masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

            return [input_ids, segment_ids, input_mask,
                    masked_lm_labels, next_sentence_labels]

    def __del__(self):
        if self.data_file:
            self.data_file.flush()
            self.data_file.close()     #encounter bug, don't know how to fix it

    def __len__(self):
        return self._len
        # debug
        # return 11

    def size(self, idx: int):
        """
        Return an example's size as a float or tuple.
        """
        return self.max_pred_length  # in our BERT preparation, the length is always 512

    def set_epoch(self, epoch):
        pass

class ConBertH5pyData(torch.utils.data.Dataset):
    @staticmethod
    def cumsum(sequence, sample_ratios):
        r, s = [], 0
        for e, ratio in zip(sequence, sample_ratios):
            curr_len = int(ratio * len(e))
            r.append(curr_len + s)
            s += curr_len
        return r

    def __init__(self, datasets, sample_ratios=1):
        super(ConBertH5pyData, self).__init__()
        assert len(datasets) > 0,  "datasets should not be an empty iterable"
        self.datasets = list(datasets)
        if isinstance(sample_ratios, int):
            sample_ratios = [sample_ratios] * len(self.datasets)
        self.sample_ratios = sample_ratios
        self.cumulative_sizes = self.cumsum(self.datasets, sample_ratios)
        self.real_sizes = [len(d) for d in self.datasets]

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(idx)
        return self.datasets[dataset_idx][sample_idx]

    def _get_dataset_and_sample_index(self, idx: int):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample_idx = sample_idx % self.real_sizes[dataset_idx]
        return dataset_idx, sample_idx

    def collater(self, samples):
        # For now only supports datasets with same underlying collater implementations
        # print("samples", type(samples))
        if len(samples) == 0:
            return None
        if hasattr(self.datasets[0], 'collater'):
            return self.datasets[0].collater(samples)
        else:
            return default_collate(samples)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return np.arange(len(self))

    def num_tokens(self, index: int):
        return np.max(self.size(index))
    
    def size(self, idx: int):
        """
        Return an example's size as a float or tuple.
        """
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(idx)
        return self.datasets[dataset_idx].size(sample_idx)

    def set_epoch(self, epoch):
        pass


