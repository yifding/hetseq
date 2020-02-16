import os
import collections

import h5py
from tqdm import tqdm

import numpy as np
import torch.utils.data


# # method is too large(memory 7% took 20GB)  and ~25min to load data.
class CombineBertData(torch.utils.data.Dataset):
    def __init__(self, files, max_pred_length=512,
                 keys=('input_ids', 'input_mask', 'segment_ids',
                       'masked_lm_positions', 'masked_lm_ids','next_sentence_labels')):
        # can potentially using multiple processing/thread to speed up reading
        # if input is too large, considering other strategies to load data
        self.max_pred_length = max_pred_length
        self.keys = keys
        self.inputs = collections.OrderedDict()

        for key in self.keys:
            self.inputs[key] = []

        for input_file in tqdm(files):
                f = h5py.File(input_file, "r")
                for i, key in enumerate(keys):
                    if i < 5:
                        self.inputs[key].append(f[key][:])
                    else:
                        self.inputs[key].append(np.asarray(f[key][:]))
                f.close()

        for key in self.inputs:
            self.inputs[key] = np.concatenate(self.inputs[key])

    def __len__(self):
        return len(self.inputs[self.keys[0]])

    def __getitem__(self, index):
        return [self.inputs[key][index] for key in self.keys]
        # ['input_ids', 'input_mask', 'segment_ids','masked_lm_positions',
        # 'masked_lm_ids','next_sentence_labels']


class BertTask(object):
    def __init__(self, args):
        self.args = self.load()
        self.dict = self.load_vocab(args.dict)
        self.seed = args.seed
        self.datasets = {}

    @staticmethod
    def load_vocab(vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        index = 0
        with open(vocab_file, "r", encoding="utf-8") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab

    def load_dataset(self, split='train'):
        """combine multiple files into one single dataset
        Args:
            split (str): name must included in the file(e.g., train, valid, test)

        """
        path = self.args.data
        if not os.path.exists(path):
            raise FileNotFoundError(
                "Dataset not found: ({})".format(path)
            )

        files = os.listdir(path) if os.path.isdir(path) else [path]
        files = [f for f in files if split in f]
        assert len(files) > 0

        self.datasets[split] = CombineBertData(files)


        """
        dataset = data_utils.load_indexed_dataset(
            split_path, self.dictionary, self.args.dataset_impl, combine=combine
        )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample,
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            break_mode=self.args.sample_break_mode,
            include_targets=True,
        )

        add_eos_for_other_targets = (
            self.args.sample_break_mode is not None
            and self.args.sample_break_mode != "none"
        )

        self.datasets[split] = MonolingualDataset(
            dataset,
            dataset.sizes,
            self.dictionary,
            self.output_dictionary,
            add_eos_for_other_targets=add_eos_for_other_targets,
            shuffle=True,
            targets=self.targets,
            add_bos_token=self.args.add_bos_token,
        )
        """