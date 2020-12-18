from collections import OrderedDict
import contextlib
from itertools import chain
import math
import os
import sys

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from hetseq import (
    utils,
    optim,
    lr_scheduler,
    checkpoint_utils,
    distributed_utils,
)

from hetseq.meters import AverageMeter, StopwatchMeter, TimeMeter


class Controller(object):
    """Main class for data parallel training.
    This class supports synchronous distributed data parallel training,
    where multiple workers each have a full model replica and gradients
    are accumulated across workers before each update. We use
    :class:`~torch.nn.parallel.DistributedDataParallel` to handle
    communication of the gradients across workers.
    """

    def __init__(self, args, task, model, criterion=None, dummy_batch=None, oom_batch=None):
        self.args = args
        self.task = task

        # copy model and criterion to current device
        self._model = model
        self.cuda = torch.cuda.is_available() and not args.cpu
        if self.cuda:
            self._model = self._model.cuda()

        self._dummy_batch = dummy_batch
        self._oom_batch = oom_batch or dummy_batch

        self._lr_scheduler = None
        self._num_updates = 0
        self._optim_history = None
        self._optimizer = None
        self._prev_grad_norm = None
        self._wrapped_criterion = None
        self._wrapped_model = None

        # Fast stats sync avoids memcpy and is 7% faster when tested on 16 nodes.
        # It is less flexible and syncs only the default stats.
        self._all_reduce_list = [0.0] * 6
        self.fast_stat_sync = args.fast_stat_sync

        self.init_meters(args)

    def init_meters(self, args):
        self.meters = OrderedDict()
        self.meters['train_loss'] = AverageMeter()
        self.meters['train_nll_loss'] = AverageMeter()
        self.meters['valid_loss'] = AverageMeter()
        self.meters['valid_nll_loss'] = AverageMeter()
        self.meters['wps'] = TimeMeter()       # words per second
        self.meters['ups'] = TimeMeter()       # updates per second
        self.meters['wpb'] = AverageMeter()    # words per batch
        self.meters['bsz'] = AverageMeter()    # sentences per batch
        self.meters['gnorm'] = AverageMeter()  # gradient norm
        self.meters['clip'] = AverageMeter()   # % of updates clipped
        self.meters['wall'] = TimeMeter()      # wall time in seconds
        self.meters['train_wall'] = StopwatchMeter()  # train wall time in seconds


    @property
    def model(self):
        if self._wrapped_model is None:
            if self.args.distributed_world_size > 1 and not self.args.use_bmuf:
                self._wrapped_model = DDP(
                    module=self._model,
                    device_ids=[self.args.device_id],
                    output_device=self.args.device_id,
                    broadcast_buffers=False,
                    bucket_cap_mb=self.args.bucket_cap_mb,
                    check_reduction=True,
                    find_unused_parameters=self.args.find_unused_parameters,
                )
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer

    @property
    def lr_scheduler(self):
        if self._lr_scheduler is None:
            self._build_optimizer()  # this will initialize self._lr_scheduler
        return self._lr_scheduler

    def _build_optimizer(self):
        params = list(
            filter(
                lambda p: p.requires_grad,
                chain(self.model.parameters()),
            )
        )

        if self.args.optimizer == 'adam':
            self._optimizer = optim._Adam(self.args, params)
        elif self.args.optimizer == 'adadelta':
            self._optimizer = optim._Adadelta(self.args, params)
        else:
            raise ValueError("unsupported optimizer - {}".format(self.args.optimizer))

        # We should initialize the learning rate scheduler immediately after
        # building the optimizer, so that the initial learning rate is set.

        if self.args.lr_scheduler == 'PolynomialDecayScheduler':
            self._lr_scheduler = lr_scheduler.PolynomialDecayScheduler(self.args, self.optimizer)
        else:
            raise ValueError("unsupported lr_scheduler - {}".format(self.args.lr_scheduler))

        self._lr_scheduler.step_update(0)

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        if distributed_utils.is_master(self.args):  # only save one checkpoint
            extra_state['train_meters'] = self.meters
            checkpoint_utils.save_state(
                filename, self.args, self.get_model().state_dict(), None,
                self.optimizer, self.lr_scheduler, self.get_num_updates(),
                self._optim_history, extra_state,
            )

    def load_checkpoint(
        self,
        filename,
        reset_optimizer=False,
        reset_lr_scheduler=False,
        optimizer_overrides=None,
        reset_meters=False,
    ):
        """Load all training state from a checkpoint file."""
        extra_state, self._optim_history, last_optim_state = None, [], None

        if os.path.exists(filename):
            state = checkpoint_utils.load_checkpoint_to_cpu(filename)

            # load model parameters
            try:
                self.get_model().load_state_dict(state['model'], strict=True)

            except Exception:
                raise Exception(
                    'Cannot load model parameters from checkpoint {}; '
                    'please ensure that the architectures match.'.format(filename)
                )

            extra_state = state['extra_state']
            self._optim_history = state['optimizer_history']
            last_optim_state = state.get('last_optimizer_state', None)

        if last_optim_state is not None and not reset_optimizer:
            # rebuild optimizer after loading model, since params may have changed
            self._build_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]

            assert last_optim['optimizer_name'] == self.optimizer.__class__.__name__, \
                'Optimizer does not match; please reset the optimizer (--reset-optimizer).'

            if not reset_lr_scheduler:
                self.lr_scheduler.load_state_dict(last_optim['lr_scheduler_state'])
            self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)

            self.set_num_updates(last_optim['num_updates'])

        if extra_state is not None:
            epoch = extra_state['train_iterator']['epoch']
            print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                filename, epoch, self.get_num_updates()))

            self.lr_step(epoch)

            if 'train_meters' in extra_state and not reset_meters:
                self.meters.update(extra_state['train_meters'])
                del extra_state['train_meters']

                # reset TimeMeters, since their start times don't make sense anymore
                for meter in self.meters.values():
                    if isinstance(meter, TimeMeter):
                        meter.reset()
        else:
            print('| no existing checkpoint found {}'.format(filename))

        return extra_state

    def get_train_iterator(self, epoch, combine=True, load_dataset=True):
        """Return an EpochBatchIterator over the training set for a given epoch."""
        if load_dataset:
            print('| loading train data for epoch {}'.format(epoch))
            self.task.load_dataset(self.args.train_subset)
        return self.task.get_batch_iterator(
            dataset=self.task.dataset(self.args.train_subset),
            max_tokens=self.args.max_tokens,
            max_sentences=self.args.max_sentences,
            max_positions=None,
            ignore_invalid_inputs=True,
            required_batch_size_multiple=self.args.required_batch_size_multiple,
            seed=self.args.seed,
            num_shards=self.args.distributed_world_size,
            shard_id=self.args.distributed_rank,
            num_workers=self.args.num_workers,
            epoch=epoch,
        )

    def train_step(self, samples, dummy_batch=False, raise_oom=False):
        """Do forward, backward and parameter update."""
        if self._dummy_batch is None:
            self._dummy_batch = samples[0]

        self._set_seed()
        self.model.train()
        self.zero_grad()

        if not dummy_batch:
            self.meters['train_wall'].start()

        # forward and backward pass
        logging_outputs, sample_sizes, ooms = [], [], 0
        for i, sample in enumerate(samples):
            sample = self._prepare_sample(sample)
            if sample is None:
                # when sample is None, run forward/backward on a dummy batch
                # and ignore the resulting gradients
                sample = self._prepare_sample(self._dummy_batch)
                ignore_grad = True
            else:
                ignore_grad = False

            def maybe_no_sync():
                """
                Whenever *samples* contains more than one mini-batch, we
                want to accumulate gradients locally and only call
                all-reduce in the last backwards pass.
                """
                if (
                    self.args.distributed_world_size > 1
                    and hasattr(self.model, 'no_sync')
                    and i < len(samples) - 1
                ):
                    return self.model.no_sync()
                else:
                    return contextlib.ExitStack()  # dummy contextmanager

            try:
                with maybe_no_sync():
                    # forward and backward

                    loss, sample_size, logging_output = self.task.train_step(
                        sample, self.model, self.optimizer,
                        ignore_grad,
                    )

                if not ignore_grad:
                    logging_outputs.append(logging_output)
                    sample_sizes.append(sample_size)

                    if self.fast_stat_sync:
                        self._all_reduce_list[0] += sample_size
                        self._all_reduce_list[1] += logging_output.get('nsentences', 0.0)
                        self._all_reduce_list[2] += logging_output.get('loss', 0.0)
                        self._all_reduce_list[3] += logging_output.get('nll_loss', 0.0)
                        self._all_reduce_list[4] += logging_output.get('ntokens', 0.0)

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    raise RuntimeError('ran out of memory with exception')

                else:
                    raise e


        if dummy_batch:
            return None

        # gather logging outputs from all replicas
        if self.fast_stat_sync:
            # rework all_gather_list
            all_reduce_list_tensor = torch.cuda.DoubleTensor(self._all_reduce_list)
            if self._sync_stats():
                torch.distributed.all_reduce(all_reduce_list_tensor)
            # Normalize loss and nll_loss by "sample_size"
            # and convert to log base 2
            all_reduce_list_tensor[2:4].div_(
                (
                    all_reduce_list_tensor[0:1] *
                    torch.log(torch.cuda.DoubleTensor([2]))
                )
            )
            self._all_reduce_list = all_reduce_list_tensor.tolist()
            logging_output = {}
            [
                sample_size,
                logging_output['nsentences'],
                logging_output['loss'],
                logging_output['nll_loss'],
                logging_output['ntokens'],
                ooms,
            ] = self._all_reduce_list
        elif self._sync_stats():
            logging_outputs, sample_sizes, ooms, prev_norms = \
                zip(*distributed_utils.all_gather_list(
                    [logging_outputs, sample_sizes, ooms, self._prev_grad_norm],
                ))
            logging_outputs = list(chain.from_iterable(logging_outputs))
            sample_sizes = list(chain.from_iterable(sample_sizes))
            ooms = sum(ooms)

            if not self.args.use_bmuf:
                assert (
                    all(norm == prev_norms[0] for norm in prev_norms)
                    or all(math.isnan(norm) or math.isinf(norm) for norm in prev_norms)
                ), 'Fatal error: gradients are inconsistent between workers'

        if not all(k in logging_output for k in ['ntokens', 'nsentences']):
            raise Exception((
                'Please update the {}.aggregate_logging_outputs() method to '
                'return ntokens and nsentences'
            ).format(self.task.__class__.__name__))

        try:
            # normalize grads by sample size
            if sample_size > 0:
                self.optimizer.multiply_grads(self.args.distributed_world_size / float(sample_size))

            # clip grads
            grad_norm = self.optimizer.clip_grad_norm(self.args.clip_norm)
            self._prev_grad_norm = grad_norm

            # take an optimization step
            self.optimizer.step()
            self.set_num_updates(self.get_num_updates() + 1)

            # task specific update per step
            self.task.update_step(self._num_updates)

            # update meters
            ntokens = logging_output.get('ntokens', 0)
            nsentences = logging_output.get('nsentences', 0)
            self.meters['wps'].update(ntokens)
            self.meters['ups'].update(1.)
            self.meters['wpb'].update(ntokens)
            self.meters['bsz'].update(nsentences)
            self.meters['gnorm'].update(grad_norm)
            self.meters['clip'].update(
                1. if grad_norm > self.args.clip_norm and self.args.clip_norm > 0 else 0.
            )
            self.meters['train_loss'].update(logging_output.get('loss', 0), sample_size)
            if 'train_acc' in self.meters:
                self.meters['train_acc'].update(
                    logging_output.get('acc', 0), sample_size)

        except OverflowError as e:
            print('| WARNING: overflow detected, ' + str(e))
            self.zero_grad()
            logging_output = None

        self.clear_buffered_stats()
        self.meters['train_wall'].stop()

        return logging_output

    def zero_grad(self):
        self.optimizer.zero_grad()

    def clear_buffered_stats(self):
        self._all_reduce_list = [0.0] * 6

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate based on the validation loss."""
        self.lr_scheduler.step(epoch, val_loss)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update()

    def lr_step_update(self):
        """Update the learning rate after each update."""      # #TODO: lr_scheduler
        return self.lr_scheduler.step_update(self.get_num_updates())

    def get_lr(self):
        """Get the current learning rate."""        # #TODO: optimizer
        return self.optimizer.get_lr()

    def get_model(self):
        """Get the (non-wrapped) model instance."""
        return self._model

    def get_meter(self, name):
        """Get a specific meter by name."""
        if name not in self.meters:
            return None
        return self.meters[name]

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self._num_updates = num_updates
        self.lr_step_update()

    def _prepare_sample(self, sample):
        if sample is None or len(sample) == 0:
            return None

        if self.cuda:
            sample = utils.move_to_cuda(sample)

        return sample

    def _set_seed(self):
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.args.seed + self.get_num_updates()
        torch.manual_seed(seed)
        if self.cuda:
            torch.cuda.manual_seed(seed)

    def _sync_stats(self):

        return (
                self.args.distributed_world_size > 1
        )

