import os
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from hetseq import distributed_utils
from hetseq.data import (
    BertNerDataset,
    MNISTDataset,
    BertH5pyData,
    ConBertH5pyData,
    data_utils,
    iterators,
)



class Task(object):
    """
    Tasks store dictionaries and provide helpers for loading/iterating over
    Datasets, initializing the Model/Criterion and calculating the loss.
    """
    def __init__(self, args):
        self.args = args
        self.datasets = {}
        self.dataset_to_epoch_iter = {}

    def load_dictionary(self, vocab_file):
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
        print('| loaded dictionary with {} subwords  from: {}'.format(index, vocab_file))
        return vocab

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        raise NotImplementedError

    def dataset(self, split):
        """
        Return a loaded dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        Returns:
            a :class:`~torch.utils.data.Dataset` corresponding to *split*
        """
        if split not in self.datasets:
            raise KeyError('Dataset not loaded: ' + split)
        if not isinstance(self.datasets[split], torch.utils.data.Dataset):
            raise TypeError('Datasets are expected to be of type torch.utils.data.Dataset')
        return self.datasets[split]

    def get_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=0,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.
        Args:
            dataset (~torch.utils.data.Dataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1). 19940802 by options default
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1) = num of gpus
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0). = index of gpu
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0). 16 by options default
            epoch (int, optional): the epoch to start the iterator from
                (default: 0).
        Returns:
            ~iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        # For default fairseq task, return same iterator across epochs
        # as datasets are not dynamic, can be overridden in task specific
        # setting.
        if dataset in self.dataset_to_epoch_iter:
            return self.dataset_to_epoch_iter[dataset]

        # initialize the dataset with the correct starting epoch
        #dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # create mini-batches with given size constraints
        print('| build batch sampler')
        batch_sampler = data_utils.batch_by_size(
            indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )
        print('| finish building batch sampler')

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )
        self.dataset_to_epoch_iter[dataset] = epoch_iter
        return epoch_iter

    def build_model(self, args):
        """
        Build the :class:`~torch.nn.Module` instance for this
        task.
        Args:
            args (argparse.Namespace): parsed command-line arguments
        Returns:
            a :class:`~torch.nn.Module` instance
        """
        raise NotImplementedError

    def train_step(self, sample, model, optimizer, ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.
        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~torch.utils.data.Dataset`.
            model (~torch.nn.Module): the model
            optimizer (~optim._Optimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True
        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()

        loss = model(*sample)
        if ignore_grad:
            loss *= 0
        if sample is None or len(sample) == 0 or len(sample[0][0]) == 0:
            sample_size = 0
        else:
            sample_size = len(sample[0][0])

        nsentences = sample_size

        logging_output = {
            'nsentences': nsentences,
            'loss': loss.data,
            'nll_loss': loss.data,
            'ntokens': 0,
            'sample_size': sample_size,
        }

        optimizer.backward(loss)
        return loss, sample_size, logging_output


    def update_step(self, num_updates):
        """Task level update when number of update increases. This is called after optimization step and
           learning rate update of each step"""
        pass


class LanguageModelingTask(Task):
    """
    Train a language model, currently support BERT.
    Args:
        args: parsed from command line
        dictionary: the BPE dictionary for the input of the language model
    """

    def __init__(self, args, dictionary):
        super(LanguageModelingTask, self).__init__(args)
        self.dictionary = dictionary
        '''
        self.output_dictionary = output_dictionary or dictionary

        if targets is None:
            targets = ["future"]
        self.targets = targets
        '''


    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).
        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        dictionary = cls.load_dictionary(cls, args.dict)

        return cls(args, dictionary)

    def build_model(self, args):
        if args.task == 'bert':
            from hetseq.bert_modeling import BertForPreTraining, BertConfig
            config = BertConfig.from_json_file(args.config_file)
            model = BertForPreTraining(config)

        else:
            raise ValueError(
                    "Unsupported language modeling task: {}".format(args.task)
                )

        return model

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        path = self.args.data
        if not os.path.exists(path):
            raise FileNotFoundError(
                "Dataset not found: ({})".format(path)
            )

        files = [os.path.join(path, f) for f in os.listdir(path)] if os.path.isdir(path) else [path]
        files = sorted([f for f in files if split in f])

        # # debug
        if self.args.num_file > 0:
            files = files[0:self.args.num_file]

        assert len(files) > 0, "no suitable file in split ***{}***".format(split)

        datasets = []
        for i, f in enumerate(files):
            datasets.append(BertH5pyData(f))

        dataset = ConBertH5pyData(datasets)

        print('| loaded {} sentences from: {}'.format(len(dataset), path), flush=True)

        self.datasets[split] = dataset
        print('| loading finished')

class MNISTTask(Task):
    def __init__(self, args):
        super(MNISTTask, self).__init__(args)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).
        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        # return cls(args, dictionary, output_dictionary, targets=targets)
        return cls(args)

    def build_model(self, args):
        model = MNISTNet()
        return model

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        path = self.args.data

        if not os.path.exists(path):
            os.makedirs(path)
            raise FileNotFoundError(
                "Dataset not found: ({})".format(path)
            )

        if os.path.isdir(path):
            if os.path.exists(os.path.join(path, 'MNIST/processed/')):
                path = os.path.join(path, 'MNIST/processed/')
            elif os.path.basename(os.path.normpath(path)) != 'processed':
                torchvision.datasets.MNIST(path, train=True, download=True)
                path = os.path.join(path, 'MNIST/processed/')

        files = [os.path.join(path, f) for f in os.listdir(path)] if os.path.isdir(path) else [path]
        files = sorted([f for f in files if split in f])

        assert len(files) == 1, "no suitable file in split ***{}***".format(split)

        dataset = MNISTDataset(files[0])

        print('| loaded {} sentences from: {}'.format(len(dataset), path), flush=True)

        self.datasets[split] = dataset
        print('| loading finished')

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, target, eval=False):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        loss = F.nll_loss(output, target)
        return loss
