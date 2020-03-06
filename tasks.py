import os
import collections

import numpy as np
import torch

'''
#from fairseq import tokenizer
from Distributed_BERT.data import data_utils, FairseqDataset, iterators, Dictionary
'''
from data import BertH5pyData, ConBertH5pyData, data_utils, iterators


class Task(object):
    """
    Tasks store dictionaries and provide helpers for loading/iterating over
    Datasets, initializing the Model/Criterion and calculating the loss.
    """

    '''
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        pass
    '''
    def __init__(self, args):
        self.args = args
        self.datasets = {}
        self.dataset_to_epoch_iter = {}

    '''
    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename
        Args:
            filename (str): the filename
        """
        return Dictionary.load(filename)
    '''

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

    '''
    @classmethod
    def build_dictionary(cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        """Build the dictionary
        Args:
            filenames (list): list of filenames
            workers (int): number of concurrent workers
            threshold (int): defines the minimum word count
            nwords (int): defines the total number of words in the final dictionary,
                including special symbols
            padding_factor (int): can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        d = Dictionary()
        for filename in filenames:
            Dictionary.add_file_to_dictionary(filename, d, tokenizer.tokenize_line, workers)
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d
    '''

    '''
    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).
        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        return cls(args, **kwargs)
    '''

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
            a :class:`~fairseq.data.FairseqDataset` corresponding to *split*
        """
        from fairseq.data import FairseqDataset
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
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
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

        '''
        # filter examples that are too large
        if max_positions is not None:
            indices = data_utils.filter_by_size(
                indices, dataset, max_positions, raise_exception=(not ignore_invalid_inputs),
            )
        '''

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
        Build the :class:`~fairseq.models.BaseFairseqModel` instance for this
        task.
        Args:
            args (argparse.Namespace): parsed command-line arguments
        Returns:
            a :class:`~fairseq.models.BaseFairseqModel` instance
        """
        '''
        from fairseq import models
        return models.build_model(args, self)
        '''
        raise NotImplementedError


    '''
    def build_criterion(self, args):
        """
        Build the :class:`~fairseq.criterions.FairseqCriterion` instance for
        this task.
        Args:
            args (argparse.Namespace): parsed command-line arguments
        Returns:
            a :class:`~fairseq.criterions.FairseqCriterion` instance
        """
        from fairseq import criterions
        return criterions.build_criterion(args, self)
    '''

    '''
    def build_generator(self, args):
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(self.target_dictionary)
        else:
            from fairseq.sequence_generator import SequenceGenerator
            return SequenceGenerator(
                self.target_dictionary,
                beam_size=getattr(args, 'beam', 5),
                max_len_a=getattr(args, 'max_len_a', 0),
                max_len_b=getattr(args, 'max_len_b', 200),
                min_len=getattr(args, 'min_len', 1),
                normalize_scores=(not getattr(args, 'unnormalized', False)),
                len_penalty=getattr(args, 'lenpen', 1),
                unk_penalty=getattr(args, 'unkpen', 0),
                sampling=getattr(args, 'sampling', False),
                sampling_topk=getattr(args, 'sampling_topk', -1),
                sampling_topp=getattr(args, 'sampling_topp', -1.0),
                temperature=getattr(args, 'temperature', 1.),
                diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
                diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
                match_source_len=getattr(args, 'match_source_len', False),
                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
            )
    '''

    '''
    # ORI:
    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.
        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True
        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output
    '''

    def train_step(self, sample, model, optimizer, ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.
        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True
        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        # ori: loss, sample_size, logging_output = criterion(model, sample)
        # cheat by adding bert parameters

        # TODO sample_size
        # TODO logging_output
        '''
        [input_ids, input_mask, segment_ids,
         masked_lm_positions, masked_lm_ids, next_sentence_labels] = sample
        '''
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

    '''
    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output
    '''

    '''
    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            return generator.generate(models, sample, prefix_tokens=prefix_tokens)
    '''

    def update_step(self, num_updates):
        """Task level update when number of update increases. This is called after optimization step and
           learning rate update of each step"""
        pass


    '''
    def grad_denom(self, sample_sizes, criterion):
        return criterion.__class__.grad_denom(sample_sizes)
    '''

    '''
    def aggregate_logging_outputs(self, logging_outputs, criterion):
        return criterion.__class__.aggregate_logging_outputs(logging_outputs)
    '''

    '''
    def max_positions(self):
        """Return the max input length allowed by the task."""
        return None
    '''

    '''
    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        raise NotImplementedError
    '''

    '''
    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        raise NotImplementedError
    '''


class LanguageModelingTask(Task):
    """
    Train a language model.
    Args:
        dictionary (~fairseq.data.Dictionary): the dictionary for the input of
            the language model
        output_dictionary (~fairseq.data.Dictionary): the dictionary for the
            output of the language model. In most cases it will be the same as
            *dictionary*, but could possibly be a more limited version of the
            dictionary (if ``--output-dictionary-size`` is used).
        targets (List[str]): list of the target types that the language model
            should predict.  Can be one of "self", "future", and "past".
            Defaults to "future".
    .. note::
        The language modeling task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate`, :mod:`fairseq-interactive` and
        :mod:`fairseq-eval-lm`.
    The language modeling task provides the following additional command-line
    arguments:
    .. argparse::
        :ref: fairseq.tasks.language_modeling_parser
        :prog:
    """
    '''
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='path to data directory')
        parser.add_argument('--sample-break-mode', default='none',
                            choices=['none', 'complete', 'complete_doc', 'eos'],
                            help='If omitted or "none", fills each sample with tokens-per-sample '
                                 'tokens. If set to "complete", splits samples only at the end '
                                 'of sentence, but may include multiple sentences per sample. '
                                 '"complete_doc" is similar but respects doc boundaries. '
                                 'If set to "eos", includes only one sentence per sample.')
        parser.add_argument('--tokens-per-sample', default=1024, type=int,
                            help='max number of tokens per sample for LM dataset')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--output-dictionary-size', default=-1, type=int,
                            help='limit the size of output dictionary')
        parser.add_argument('--self-target', action='store_true',
                            help='include self target')
        parser.add_argument('--future-target', action='store_true',
                            help='include future target')
        parser.add_argument('--past-target', action='store_true',
                            help='include past target')
        parser.add_argument('--add-bos-token', action='store_true',
                            help='prepend beginning of sentence token (<s>)')
        parser.add_argument('--max-target-positions', type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        # fmt: on
    '''

    #def __init__(self, args, dictionary, output_dictionary=None, targets=None):
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
        '''
        if getattr(args, "raw_text", False):
            utils.deprecation_warning(
                "--raw-text is deprecated, please use --dataset-impl=raw"
            )
            args.dataset_impl = "raw"
        elif getattr(args, "lazy_load", False):
            utils.deprecation_warning(
                "--lazy-load is deprecated, please use --dataset-impl=lazy"
            )
            args.dataset_impl = "lazy"

        dictionary = None
        output_dictionary = None
        if args.data:
            paths = args.data.split(":")
            assert len(paths) > 0
            dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
            print("| dictionary: {} types".format(len(dictionary)))
            output_dictionary = dictionary
            if args.output_dictionary_size >= 0:
                output_dictionary = TruncatedDictionary(
                    dictionary, args.output_dictionary_size
                )

        # upgrade old checkpoints
        if hasattr(args, "exclude_self_target"):
            args.self_target = not args.exclude_self_target

        targets = []
        if getattr(args, "self_target", False):
            targets.append("self")
        if getattr(args, "future_target", False):
            targets.append("future")
        if getattr(args, "past_target", False):
            targets.append("past")
        if len(targets) == 0:
            # standard language modeling
            targets = ["future"]
        '''
        dictionary = cls.load_dictionary(cls, args.dict)

        #return cls(args, dictionary, output_dictionary, targets=targets)
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
        #print('| load finished')


    '''
    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return TransformEosDataset(
            MonolingualDataset(
                TokenBlockDataset(
                    src_tokens,
                    src_lengths,
                    block_size=None,
                    pad=self.source_dictionary.pad(),
                    eos=self.source_dictionary.eos(),
                    break_mode="eos",
                    include_targets=False,
                ),
                src_lengths,
                self.source_dictionary,
                self.target_dictionary,
                add_eos_for_other_targets=False,
                shuffle=False,
                add_bos_token=self.args.add_bos_token,
            ),
            eos=self.source_dictionary.eos(),
            # remove EOS since this will be used as a prefix for generation
            remove_eos_from_src=True,
            has_target=False,
        )
    '''

    '''
    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            if prefix_tokens is None and sample["net_input"]["src_tokens"].nelement():
                # note: EOS has already been removed in build_dataset_for_inference
                prefix_tokens = sample["net_input"]["src_tokens"]
            return generator.generate(models, sample, prefix_tokens=prefix_tokens)
    '''

    '''
    @property
    def source_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary
    '''

    '''
    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.output_dictionary
    '''