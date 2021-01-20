import sys
import math
import random
import argparse
import collections

import torch
import numpy as np

from tqdm import tqdm

from hetseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    progress_bar,
    tasks,
    utils
)
from hetseq.data import iterators
from hetseq.controller import Controller
from hetseq.meters import AverageMeter, StopwatchMeter


def main(args, init_distributed=False):
    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)

    #  set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    print(args, flush=True)

    # Setup task, e.g., translation, language modeling, etc.
    task = None
    if args.task == 'bert':
        task = tasks.LanguageModelingTask.setup_task(args)
    elif args.task == 'mnist':
        task = tasks.MNISTTask.setup_task(args)
    elif args.task == 'BertForTokenClassification':
        task = tasks.BertForTokenClassificationTask.setup_task(args)
    elif args.task == 'BertForELClassification':
        task = tasks.BertForELClassificationTask.setup_task(args)
    assert task != None

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    # Build model
    model = task.build_model(args)

    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Build controller
    controller = Controller(args, task, model)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator

    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, controller)

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf

    lr = controller.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()

    while (
            lr > args.min_lr
            and (epoch_itr.epoch < max_epoch
            or (epoch_itr.epoch == max_epoch
                and epoch_itr._next_epoch_itr is not None))
            and controller.get_num_updates() < max_update
    ):
        # train for one epoch
        train(args, controller, task, epoch_itr)                   # #revise-task 6

        # debug
        valid_losses = [None]

        # only use first validation loss to update the learning rate
        lr = controller.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            checkpoint_utils.save_checkpoint(args, controller, epoch_itr, valid_losses[0])

        reload_dataset = hasattr(args, 'data') and args.data is not None and ':' in getattr(args, 'data', '')
        # sharded data: get train iterator for next epoch
        epoch_itr = controller.get_train_iterator(epoch_itr.epoch, load_dataset=reload_dataset)

    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


def train(args, controller, task, epoch_itr):                  # #revise-task 7
    """Train the model for one epoch."""
    # Update parameters every N batches, CORE scaling method
    update_freq = args.update_freq[epoch_itr.epoch - 1] \
        if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )

    itr = iterators.GroupedIterator(itr, update_freq)

    progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf


    loop = enumerate(progress, start=epoch_itr.iterations_in_epoch)

    for i, samples in loop:
        log_output = controller.train_step(samples)
        if log_output is None:
            continue

        # log mid-epoch stats

        stats = get_training_stats(controller)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k or k == 'accuracy':
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats, tag='train', step=stats['num_updates'])

        # ignore the first mini-batch in words-per-second and updates-per-second calculation
        if i == 0:
            controller.get_meter('wps').reset()
            controller.get_meter('ups').reset()

        num_updates = controller.get_num_updates()

        if num_updates >= max_update:
            break


def get_training_stats(controller):
    stats = collections.OrderedDict()
    stats['loss'] = controller.get_meter('train_loss')     # #training loss
    if controller.get_meter('train_nll_loss').count > 0:
        nll_loss = controller.get_meter('train_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = controller.get_meter('train_loss')      # #null loss?
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)       # #perplexity 2**null_loss.avg
    stats['wps'] = controller.get_meter('wps')     # #words per second
    stats['ups'] = controller.get_meter('ups')     # #updates per second
    stats['wpb'] = controller.get_meter('wpb')     # #?words per batch?
    stats['bsz'] = controller.get_meter('bsz')     # #batch size
    stats['num_updates'] = controller.get_num_updates()        # #number of updates
    stats['lr'] = controller.get_lr()      # #learning rate
    stats['gnorm'] = controller.get_meter('gnorm')     # #?normalization
    stats['clip'] = controller.get_meter('clip')       # #gradient clip
    stats['oom'] = controller.get_meter('oom')         # #out of memory
    if controller.get_meter('loss_scale') is not None:
        stats['loss_scale'] = controller.get_meter('loss_scale')
    stats['wall'] = round(controller.get_meter('wall').elapsed_time)       # #walk time
    stats['train_wall'] = controller.get_meter('train_wall')       # #training walk time
    return stats


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def cli_main():
    task_parser = argparse.ArgumentParser(allow_abbrev=False)
    task_parser.add_argument('--task', type=str,
                        default='bert', choices=['bert', 'mnist', 'BertForELClassification', 'BertForTokenClassification'])
    task_parser.add_argument('--optimizer', type=str,
                             default='adam', choices=['adam', 'adadelta'])
    task_parser.add_argument('--lr-scheduler', type=str,
                             default='PolynomialDecayScheduler', choices=['PolynomialDecayScheduler'])

    pre_args, s = task_parser.parse_known_args()

    parser = options.get_training_parser(task=pre_args.task,
                                         optimizer=pre_args.optimizer,
                                         lr_scheduler=pre_args.lr_scheduler,
                                         )
    args = options.parse_args_and_arch(parser, s)

    if args.distributed_init_method is not None:
        assert args.distributed_gpus <= torch.cuda.device_count()

        if args.distributed_gpus > 1 and not args.distributed_no_spawn:     # #by default run this logic
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=args.distributed_gpus,
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)


if __name__ == "__main__":
    cli_main()