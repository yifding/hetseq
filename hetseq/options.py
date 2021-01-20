import torch
import argparse


def get_training_parser(task='bert', optimizer='adam', lr_scheduler='PolynomialDecayScheduler'):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--no-progress-bar', action='store_true', help='disable progress bar')
    parser.add_argument('--seed', default=19940802, type=int, metavar='N',
                        help='pseudo random number generator seed')
    parser.add_argument('--cpu', action='store_true', help='use CPU instead of CUDA')
    # parser.add_argument('--fp16', action='store_true', help='use FP16')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='log progress every N batches (when progress bar is disabled)')
    parser.add_argument('--log-format', default='simple',
                        help='log format to use',
                        choices=['none', 'simple'],)

    add_dataset_args(parser, train=True, task=task)    # data file, directory and etc.
    add_distributed_training_args(parser)   # number of nodes, gpus, communication

    # initial & stop LR, update frequency, stop epoch, stop updates
    add_optimization_args(parser, optimizer=optimizer, lr_scheduler=lr_scheduler)
    add_checkpoint_args(parser)

    return parser


def add_dataset_args(parser, train=False, gen=False,  task='bert'):
    group = parser.add_argument_group('Dataset and data loading')

    group.add_argument('--num-workers', default=-1, type=int, metavar='N',
                       help='how many subprocesses to use for data loading')
    group.add_argument('--max-tokens', type=int, metavar='N',
                       help='maximum number of tokens in a batch')
    group.add_argument('--max-sentences', '--batch-size', type=int, metavar='N',
                       help='maximum number of sentences in a batch')
    group.add_argument('--required-batch-size-multiple', default=1, type=int, metavar='N',
                       help='batch size will be a multiplier of this value')

    if train:
        group.add_argument('--train-subset', default='train', metavar='SPLIT',
                           choices=['train', 'valid', 'test'],
                           help='data subset to use for training (train, valid, test)')
        group.add_argument('--valid-subset', default='valid', metavar='SPLIT',
                           help='comma separated list of data subsets to use for validation'
                                ' (train, valid, valid1, test, test1)')
        group.add_argument('--validate-interval', type=int, default=1, metavar='N',
                           help='validate every N epochs')

        group.add_argument('--disable-validation', action='store_true',
                           help='disable validation')
        group.add_argument('--max-tokens-valid', type=int, metavar='N',
                           help='maximum number of tokens in a validation batch'
                                ' (defaults to --max-tokens)')
        group.add_argument('--max-sentences-valid', type=int, metavar='N',
                           help='maximum number of sentences in a validation batch'
                                ' (defaults to --max-sentences)')
        group.add_argument('--curriculum', default=0, type=int, metavar='N',
                           help='don\'t shuffle batches for first N epochs')

        if task == 'bert':
            # personal adding for BERT data
            parser.add_argument('--task', type=str,
                                default='bert')
            parser.add_argument('--data', type=str,
                                help='path including data')
            group.add_argument('--dict', type=str, metavar='PATH of a file',
                               help='PATH to dictionary')
            group.add_argument('--config_file', type=str, metavar='PATH of a file',
                               help='PATH to bert model configuration', required=True)
            group.add_argument('--max_pred_length', type=int, default=512,
                               help='max number of tokens in a sentence')

            group.add_argument('--num_file', type=int, default=0,
                               help='number of file to run, 0 for all')

        elif task == 'mnist':
            parser.add_argument('--task', type=str, default='mnist')
            parser.add_argument('--data', type=str,
                                help='path including data')

        elif task == 'BertForTokenClassification':
            parser.add_argument('--task', type=str, default='BertForTokenClassification')
            parser.add_argument('--data', type=str,
                                help='path including data')

            group.add_argument('--dict', type=str, metavar='PATH of a file',
                               help='PATH to dictionary')
            group.add_argument('--config_file', type=str, metavar='PATH of a file',
                               help='PATH to bert model configuration', required=True)
            group.add_argument('--max_pred_length', type=int, default=512,
                               help='max number of tokens in a sentence')

            group.add_argument('--hetseq_state_dict', type=str, default=None,
                               help='PATH to load hetseq model state dictionary')
            group.add_argument('--transformers_state_dict', type=str, default=None,
                               help='PATH to load transformers official model state dictionary')

            group.add_argument('--train_file', type=str, default=None,
                               help='PATH to training file')
            group.add_argument('--validation_file', type=str, default=None,
                               help='PATH to validation file')
            group.add_argument('--test_file', type=str, default=None,
                               help='PATH to test file')
            group.add_argument('--extension_file', type=str, default=None,
                               help='PATH to extension file to build NER datasets')

            """ **YD** obtain by reading the NER data
            group.add_argument('--num_label', type=int, default=3,
                               help='Number of labels in NER output')
            """

            group.add_argument('--load_state_dict_strict', type=eval,
                               default="False",
                               help='whether strictly load state_dict')

        elif task == 'BertForELClassification':
            parser.add_argument('--task', type=str, default='BertForELClassification')
            parser.add_argument('--data', type=str,
                                help='path including data')

            group.add_argument('--dict', type=str, metavar='PATH of a file',
                               help='PATH to dictionary')
            group.add_argument('--config_file', type=str, metavar='PATH of a file',
                               help='PATH to bert model configuration', required=True)
            group.add_argument('--max_pred_length', type=int, default=512,
                               help='max number of tokens in a sentence')

            group.add_argument('--hetseq_state_dict', type=str, default=None,
                               help='PATH to load hetseq model state dictionary')
            group.add_argument('--transformers_state_dict', type=str, default=None,
                               help='PATH to load transformers official model state dictionary')

            group.add_argument('--train_file', type=str, default=None,
                               help='PATH to training file')
            group.add_argument('--validation_file', type=str, default=None,
                               help='PATH to validation file')
            group.add_argument('--test_file', type=str, default=None,
                               help='PATH to test file')
            group.add_argument('--extension_file', type=str, default=None,
                               help='PATH to extension file to build NER datasets')

            """ **YD** obtain by reading the NER data
            group.add_argument('--num_label', type=int, default=3,
                               help='Number of labels in NER output')
            """

            group.add_argument('--load_state_dict_strict', type=eval,
                               default="False",
                               help='whether strictly load state_dict')


            # **YD** deep_ed arguments
            parser.add_argument(
                '--root_data_dir',
                type=str,
                default='/home/yding4/EL_resource/data/deep_ed_PyTorch_data/',
                help='Root path of the data, $DATA_PATH.',
            )

            parser.add_argument(
                '--entities',
                type=str,
                default='RLTD',
                choices=['RLTD', '4EX', 'ALL'],
                help='Set of entities for which we train embeddings: 4EX (tiny, for debug) |'
                     ' RLTD (restricted set) | ALL (all Wiki entities, too big to fit on a single GPU)',
            )

            parser.add_argument(
                '--ent_vecs_filename',
                type=str,
                default='/home/yding4/EL_resource/data/deep_ed_PyTorch_data/generated/ent_vecs/ent_vecs__ep_9.pt',
                help='entity embedding file for given dictionary',
            )

        else:
            raise ValueError('unsupported task: {}'.format(task))


def add_distributed_training_args(parser):
    """
    Core distributed parameters interference parameters
    """

    group = parser.add_argument_group('Distributed training')

    group.add_argument('--distributed-world-size', type=int, metavar='N',
                       default=max(1, torch.cuda.device_count()),
                       help='total number of GPUs across all nodes (default: all visible GPUs)')

    group.add_argument('--distributed-rank', default=0, type=int,
                       help='rank of the current GPU')

    group.add_argument('--distributed-gpus', default=4, type=int,
                        help='number of gpus used in the current workder/node')

    group.add_argument('--distributed-backend', default='nccl', type=str,
                       help='distributed backend')

    group.add_argument('--distributed-init-method', default=None, type=str,
                       help='typically tcp://hostname:port that will be used to '
                            'establish initial connection')

    group.add_argument('--device-id', '--local_rank', default=0, type=int,
                       help='which GPU to use (usually configured automatically)')
    group.add_argument('--distributed-no-spawn', action='store_true',
                       help='do not spawn multiple processes even if multiple GPUs are visible')

    group.add_argument('--ddp-backend', default='c10d', type=str,
                       choices=['c10d'],
                       help='DistributedDataParallel backend',)

    ## utilized in the nn.parallel.DistributedDataParallel
    group.add_argument('--bucket-cap-mb', default=25, type=int, metavar='MB',
                       help='bucket size for reduction')

    group.add_argument('--fix-batches-to-gpus', action='store_true',
                       help='don\'t shuffle batches between GPUs; this reduces overall '
                            'randomness and may affect precision but avoids the cost of '
                            're-reading the data')

    ##utilized in the nn.parallel.DistributedDataParallel
    group.add_argument('--find-unused-parameters', default=False, action='store_true',
                       help='disable unused parameter detection (not applicable to '
                            'no_c10d ddp-backend')

    group.add_argument('--fast-stat-sync', default=False, action='store_true',
                       help='Enable fast sync of stats between nodes, this hardcodes to '
                            'sync only some default stats from logging_output.')
    # fmt: on
    return group


def add_optimization_args(parser, optimizer='adam', lr_scheduler='PolynomialDecayScheduler'):
    group = parser.add_argument_group('Optimization')

    group.add_argument('--max-epoch', '--me', default=0, type=int, metavar='N',
                       help='force stop training at specified epoch')
    group.add_argument('--max-update', '--mu', default=0, type=int, metavar='N',
                       help='force stop training at specified update')
    group.add_argument('--clip-norm', default=25, type=float, metavar='NORM',
                       help='clip threshold of gradients')

    group.add_argument('--update-freq', default='1', metavar='N1,N2,...,N_K',
                       type=lambda uf: eval_str_list(uf, type=int),
                       help='update parameters every N_i batches, when in epoch i')
    group.add_argument('--lr', '--learning-rate', default='0.25', type=eval_str_list,
                       metavar='LR_1,LR_2,...,LR_N',
                       help='learning rate for the first N epochs; all epochs >N using LR_N'
                            ' (note: this may be interpreted differently depending on --lr-scheduler)')
    group.add_argument('--min-lr', default=-1, type=float, metavar='LR',
                       help='stop training when the learning rate reaches this minimum')

    # # giant optimizer for all the nodes
    group.add_argument('--use-bmuf', default=False, action='store_true',
                       help='specify global optimizer for syncing models on different GPUs/shards')

    if optimizer == 'adam':
        group.add_argument('--optimizer', default='adam', type=str,
                           help='pass adam to controller to select optim class')
        group.add_argument('--adam-betas', default='(0.9, 0.999)', metavar='B',
                            help='betas for Adam optimizer')
        group.add_argument('--adam-eps', type=float, default=1e-8, metavar='D',
                            help='epsilon for Adam optimizer')
        group.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')

    elif optimizer == 'adadelta':
        group.add_argument('--optimizer', default='adadelta', type=str,
                           help='pass adam to controller to select optim class')

        group.add_argument('--adadelta_rho', default='0.9', type=float,)
        group.add_argument('--adadelta_eps', default='1e-6', type=float,)
        group.add_argument('--dadelta_weight_decay', default='0', type=float,)


    else:
        raise ValueError('unsupported optimizer: {}'.format(optimizer))

    if lr_scheduler == 'PolynomialDecayScheduler':
        group.add_argument('--lr_scheduler', default='PolynomialDecayScheduler',
                           type=str, help='pass poly lr_scheduler to controller to select optim class')

        group.add_argument('--force-anneal', '--fa', type=int, metavar='N',
                            help='force annealing at specified epoch')
        group.add_argument('--warmup-updates', default=0, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        group.add_argument('--end-learning-rate', default=0.0, type=float)
        group.add_argument('--power', default=1.0, type=float)
        group.add_argument('--total-num-update', default=1000000, type=int)

    else:
        raise ValueError('unsupported lr_scheduler: {}'.format(lr_scheduler))


    return group


def add_checkpoint_args(parser):
    group = parser.add_argument_group('Checkpointing')

    #checkpoint loading/saving directory/file
    group.add_argument('--save-dir', metavar='DIR', default='checkpoints',
                       help='path to save checkpoints')
    group.add_argument('--restore-file', default='checkpoint_last.pt',
                       help='filename from which to load checkpoint '
                            '(default: <save-dir>/checkpoint_last.pt')


    #reset parameters from checking point
    group.add_argument('--reset-dataloader', action='store_true',
                       help='if set, does not reload dataloader state from the checkpoint')
    group.add_argument('--reset-lr-scheduler', action='store_true',
                       help='if set, does not load lr scheduler state from the checkpoint')

    group.add_argument('--reset-meters', action='store_true',
                       help='if set, does not load meters from the checkpoint')

    group.add_argument('--reset-optimizer', action='store_true',
                       help='if set, does not load optimizer state from the checkpoint')

    group.add_argument('--optimizer-overrides', default="{}", type=str, metavar='DICT',
                       help='a dictionary used to override optimizer args when loading a checkpoint')

    group.add_argument('--save-interval', type=int, default=1, metavar='N',
                       help='save a checkpoint every N epochs')

    group.add_argument('--save-interval-updates', type=int, default=0, metavar='N',
                       help='save a checkpoint (and validate) every N updates')


    #checkpoints saving options
    group.add_argument('--keep-interval-updates', type=int, default=-1, metavar='N',
                       help='keep the last N checkpoints saved with --save-interval-updates')
    group.add_argument('--keep-last-epochs', type=int, default=-1, metavar='N',
                       help='keep last N epoch checkpoints')
    group.add_argument('--no-save', action='store_true',
                       help='don\'t save models or checkpoints')
    group.add_argument('--no-epoch-checkpoints', action='store_true',
                       help='only store last and best checkpoints')
    group.add_argument('--no-last-checkpoints', action='store_true',
                       help='don\'t store last checkpoints')
    group.add_argument('--no-save-optimizer-state', action='store_true',
                       help='don\'t save optimizer-state as part of checkpoint')
    group.add_argument('--best-checkpoint-metric', type=str, default='loss',
                       help='metric to use for saving "best" checkpoints')
    group.add_argument('--maximize-best-checkpoint-metric', action='store_true',
                       help='select the largest metric value for saving "best" checkpoints')
    # fmt: on

    return group


def eval_str_list(x, type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(type, x))
    except TypeError:
        return [type(x)]


def eval_bool(x, default=False):
    if x is None:
        return default
    try:
        return bool(eval(x))
    except TypeError:
        return default


def parse_args_and_arch(parser, s):
    # Post-process args.
    args = parser.parse_args(s)
    if hasattr(args, 'max_sentences_valid') and args.max_sentences_valid is None:
        args.max_sentences_valid = args.max_sentences
    if hasattr(args, 'max_tokens_valid') and args.max_tokens_valid is None:
        args.max_tokens_valid = args.max_tokens

    return args