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

_NER_COLUMNS = ['input_ids', 'labels', 'token_type_ids', 'attention_mask']


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
            from bert_modeling import BertForPreTraining, BertConfig
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


class BertForTokenClassificationTask(Task):
    def __init__(self, args):
        super(BertForTokenClassificationTask, self).__init__(args)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).
        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        # **YD** add BertTokenizerFast to be suitable for CONLL2003 NER task, pipeline is similar to
        # https://github.com/huggingface/transformers/tree/master/examples/token-classification
        # 1.obtain tokenizer and data_collator

        import datasets
        from datasets import ClassLabel
        # from transformers import BertTokenizerFast, DataCollatorForTokenClassification
        from transformers import BertTokenizerFast
        from hetseq.data_collator import YD_DataCollatorForTokenClassification

        tokenizer = BertTokenizerFast(args.dict)
        data_collator = YD_DataCollatorForTokenClassification(tokenizer, max_length=args.max_pred_length, padding=True)

        # 2. process datasets, (tokenization of NER data)
        # **YD**, add args in option.py for fine-tuning task
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        if args.test_file is not None:
            data_files["test"] = args.test_file
        extension = args.extension_file
        dataset = datasets.load_dataset(extension, data_files=data_files)

        # 3. setup num_labels
        if 'train' in dataset:
            column_names = dataset["train"].column_names
            features = dataset["train"].features
        elif 'validation' in dataset:
            column_names = dataset["validation"].column_names
            features = dataset["validation"].features
        elif 'test' in dataset:
            column_names = dataset["test"].column_names
            features = dataset["test"].features
        else:
            raise ValueError('dataset must contain "train"/"validation"/"test"')

        text_column_name = "tokens" if "tokens" in column_names else column_names[0]
        label_column_name = ('ner_tags' if 'ner_tags' in column_names else column_names[1])

        if isinstance(features[label_column_name].feature, ClassLabel):
            label_list = features[label_column_name].feature.names
            # No need to convert the labels since they are already ints.
            label_to_id = {i: i for i in range(len(label_list))}
        else:
            if 'train' in label_column_name:
                label_list = get_label_list(datasets["train"][label_column_name])
            elif 'validation' in label_column_name:
                label_list = get_label_list(datasets["validation"][label_column_name])
            elif 'test' in label_column_name:
                label_list = get_label_list(datasets["test"][label_column_name])
            else:
                raise ValueError('dataset must contain "train"/"validation"/"test"')

            label_to_id = {l: i for i, l in enumerate(label_list)}

        num_labels = len(label_list)

        # 4. tokenization
        # Tokenize all texts and align the labels with them.
        def tokenize_and_align_labels(examples, label_all_tokens=False):
            tokenized_inputs = tokenizer(
                examples[text_column_name],
                padding=False,
                truncation=True,
                # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                is_split_into_words=True,
                return_offsets_mapping=True,
            )
            offset_mappings = tokenized_inputs.pop("offset_mapping")
            labels = []
            for label, offset_mapping in zip(examples[label_column_name], offset_mappings):
                label_index = 0
                current_label = -100
                label_ids = []
                for offset in offset_mapping:
                    # We set the label for the first token of each word. Special characters will have an offset of (0, 0)
                    # so the test ignores them.
                    if offset[0] == 0 and offset[1] != 0:
                        current_label = label_to_id[label[label_index]]
                        label_index += 1
                        label_ids.append(current_label)
                    # For special tokens, we set the label to -100 so it's automatically ignored in the loss function.
                    elif offset[0] == 0 and offset[1] == 0:
                        label_ids.append(-100)
                    # For the other tokens in a word, we set the label to either the current label or -100, depending on
                    # the label_all_tokens flag.
                    else:
                        label_ids.append(current_label if label_all_tokens else -100)

                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        tokenized_datasets = dataset.map(
            tokenize_and_align_labels,
            batched=True,
            num_proc=args.num_workers,
            load_from_cache_file=False,
        )

        # 5. set up dataset format and input/output pipeline of dataset
        tokenized_datasets.set_format(type='torch', columns=_NER_COLUMNS)

        # include components in args
        args.tokenized_datasets = tokenized_datasets
        args.num_labels = num_labels
        args.tokenizer = tokenizer
        args.data_collator = data_collator

        return cls(args)

    def build_model(self, args):
        if args.task == 'BertForTokenClassification':
            # obtain num_label from dataset before assign model
            from transformers import BertForTokenClassification, BertConfig
            config = BertConfig.from_json_file(args.config_file)
            # **YD** mention detection, num_label is by default 3
            assert hasattr(args, 'num_labels')
            config.num_labels = args.num_labels
            model = BertForTokenClassification(config)

            # **YD** add load state_dict from pre-trained model
            # could make only master model to load from state_dict, not quite sure whether this works for single GPU
            # if distributed_utils.is_master(args) and args.hetseq_state_dict is not None:
            if args.hetseq_state_dict is not None:
                state_dict = torch.load(args.hetseq_state_dict, map_location='cpu')['model']
                if args.load_state_dict_strict:
                    model.load_state_dict(state_dict, strict=True)
                else:
                    model.load_state_dict(state_dict, strict=False)

            elif args.transformers_state_dict is not None:
                state_dict = torch.load(args.transformers_state_dict, map_location='cpu')
                if args.load_state_dict_strict:
                    model.load_state_dict(state_dict, strict=True)
                else:
                    model.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError('Unknown fine_tunning task!')
        return model

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        """
        assert split in ['train', 'validation', 'test']
        if split in self.datasets:
            return self.datasets[split]
        """
        if split in self.datasets:
            return

        if 'train' in self.args.tokenized_datasets:
            self.datasets['train'] = BertNerDataset(self.args.tokenized_datasets['train'], self.args)
        elif 'validation' in dataset:
            self.datasets['valid'] = BertNerDataset(self.args.tokenized_datasets['validation'], self.args)
        elif 'test' in dataset:
            self.datasets['test'] = BertNerDataset(self.args.tokenized_datasets['test'], self.args)
        else:
            raise ValueError('dataset must contain "train"/"validation"/"test"')

        print('| loading finished')

    # **YD** may need to write customized train_step, we will see
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

        loss = model(**sample)['loss']
        if ignore_grad:
            loss *= 0
        if sample is None or len(sample['labels']) == 0:
            sample_size = 0
        else:
            # sample_size = len(sample['labels'])
            sample_size = 1

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
