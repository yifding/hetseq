import math

import torch


class _Optimizer(object):
    def __init__(self, args):
        super().__init__()
        self.args = args

    '''
    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        pass
    '''

    @property
    def optimizer(self):
        """Return a torch.optim.optimizer.Optimizer instance."""
        if not hasattr(self, '_optimizer'):
            raise NotImplementedError
        if not isinstance(self._optimizer, torch.optim.Optimizer):
            raise ValueError('_optimizer must be an instance of torch.optim.Optimizer')
        return self._optimizer

    '''
    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        raise NotImplementedError
    '''

    @property
    def params(self):
        """Return an iterable of the parameters held by the optimizer."""
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                yield p

    '''
    def __getstate__(self):
        return self._optimizer.__getstate__()
    '''

    '''
    def get_lr(self):
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    '''

    '''
    def set_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    '''

    '''
    def state_dict(self):
        """Return the optimizer's state dict."""
        return self.optimizer.state_dict()
    '''

    '''
    def load_state_dict(self, state_dict, optimizer_overrides=None):
        """Load an optimizer state dict.
        In general we should prefer the configuration of the existing optimizer
        instance (e.g., learning rate) over that found in the state_dict. This
        allows us to resume training from a checkpoint using a new set of
        optimizer args.
        """
        self.optimizer.load_state_dict(state_dict)

        if optimizer_overrides is not None and len(optimizer_overrides) > 0:
            # override learning rate, momentum, etc. with latest values
            for group in self.optimizer.param_groups:
                group.update(optimizer_overrides)
    '''

    '''
    def backward(self, loss):
        """Computes the sum of gradients of the given tensor w.r.t. graph leaves."""
        loss.backward()
    '''

    '''
    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        for p in self.params:
            if p.grad is not None:
                p.grad.data.mul_(c)
    '''

    '''
    def clip_grad_norm(self, max_norm):
        """Clips gradient norm."""
        if max_norm > 0:
            return torch.nn.utils.clip_grad_norm_(self.params, max_norm)
        else:
            return math.sqrt(sum(p.grad.data.norm()**2 for p in self.params if p.grad is not None))
    '''

    def step(self, closure=None):
        """Performs a single optimization step."""
        self.optimizer.step(closure)

    '''
    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for p in self.params:
            p.grad = None
        self.optimizer.zero_grad()
    '''

    '''
    @property
    def supports_memory_efficient_fp16(self):
        if hasattr(self.optimizer, 'supports_memory_efficient_fp16'):
            return self.optimizer.supports_memory_efficient_fp16
        return False
    '''

    '''
    def average_params(self):
        pass
    '''


class _Adam(_Optimizer):
    """Adam optimizer for fairseq.
    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, args, params):
        super().__init__(args)
        '''
        if torch.cuda.is_available():
            try:
                from apex.optimizers import FusedAdam as _FusedAdam  # noqa
                self._optimizer = FusedAdam(params, **self.optimizer_config)
            except ImportError:
                self._optimizer = Adam(params, **self.optimizer_config)
        else:
            self._optimizer = Adam(params, **self.optimizer_config)
        '''
        self._optimizer = Adam(params, **self.optimizer_config)

    '''
    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--adam-betas', default='(0.9, 0.999)', metavar='B',
                            help='betas for Adam optimizer')
        parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='D',
                            help='epsilon for Adam optimizer')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        # fmt: on
    '''

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            'lr': self.args.lr[0],
            'betas': eval(self.args.adam_betas),
            'eps': self.args.adam_eps,
            'weight_decay': self.args.weight_decay,
        }

    '''
    def average_params(self):
        """Reduce Params is only used during BMUF distributed training."""
        state_dict = self.optimizer.state_dict()
        total_gpus = float(dist.get_world_size())

        for _, value in state_dict["state"].items():
            value["exp_avg"] /= total_gpus
            value["exp_avg_sq"] /= total_gpus
            dist.all_reduce(value["exp_avg"], op=dist.ReduceOp.SUM)
            dist.all_reduce(value["exp_avg_sq"], op=dist.ReduceOp.SUM)
    '''

class Adam(torch.optim.Optimizer):
    """Implements Adam algorithm.
    This implementation is modified from torch.optim.Adam based on:
    `Fixed Weight Decay Regularization in Adam`
    (see https://arxiv.org/abs/1711.05101)
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    '''
    @property
    def supports_memory_efficient_fp16(self):
        return True
    '''

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                p_data_fp32 = p.data.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                    if amsgrad:
                        state['max_exp_avg_sq'] = state['max_exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                p.data.copy_(p_data_fp32)

        return loss

'''
class BertLAMB(Optimizer):
    """Implements BERT version of LAMB algorithm.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: LAMBs b1. Default: 0.9
        b2: LAMBs b2. Default: 0.999
        e: LAMBs epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum global norm for the gradients. Default: 1.0
    """

    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_poly',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01,
                 max_grad_norm=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(BertLAMB, self).__init__(params, defaults)
        self.step_count = 0
        self.b1 = b1
        self.b2 = b2
        self.epsilon = e
        self.max_global_grad_norm = max_grad_norm
        self.learning_rate = lr
        self.schedule = schedule
        self.warmup = warmup
        self.max_steps = t_total
        self.updates_created = False

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step'] / group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def apply_gradients(self, dummy_overflow_buf, lr_scheduled, per_param_decay, grad_list, param_list, momentum,
                        velocity, update):
        # Compute global gradient norm
        global_grad_norm = multi_tensor_applier(
            multi_tensor_l2norm,
            dummy_overflow_buf,
            [grad_list],
            False)[0].item()

        # Compute per parameter norm
        param_norms = multi_tensor_applier(
            multi_tensor_l2norm,
            dummy_overflow_buf,
            [param_list],
            True)[1]

        # Compute LAMB update
        multi_tensor_applier(
            lamb_compute_update,
            dummy_overflow_buf,
            [grad_list, param_list, momentum, velocity, update],
            torch.cuda.FloatTensor(per_param_decay),
            self.step_count,
            self.b1,
            self.b2,
            self.epsilon,
            global_grad_norm,
            self.max_global_grad_norm,
        )

        # Computer per parameter update norm
        update_norms = multi_tensor_applier(
            multi_tensor_l2norm,
            dummy_overflow_buf,
            [update],
            True)[1]

        # Apply LAMB update on parameters
        multi_tensor_applier(
            lamb_apply_update,
            dummy_overflow_buf,
            [param_list, update],
            param_norms,
            update_norms,
            lr_scheduled,
        )

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        check = 1  # torch.norm(all_grads, 2)

        grad_list = []
        param_list = []
        per_param_decay = []
        momentum = []
        velocity = []

        fp16_grad_list = []
        fp16_from_fp32_param_list = []
        fp32_param_list = []
        fp16_per_param_decay = []
        fp16_momentum = []
        fp16_velocity = []

        if not self.updates_created:
            self.update = []
            self.fp16_update = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Keep step here for compatibility with earlier resume from checkpoint
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['momentum'] = torch.zeros_like(p.data, dtype=torch.float32)
                    # Exponential moving average of squared gradient values
                    state['velocity'] = torch.zeros_like(p.data, dtype=torch.float32)
                    # fp32 master weights
                if 'master_param' not in state.keys() and p.type() == 'torch.cuda.HalfTensor':
                    state['master_param'] = p.detach().clone().float()

                # ensure these 3 are float tensors
                if state['momentum'].type() != 'torch.cuda.FloatTensor':
                    state['momentum'] = state['momentum'].float()
                if state['velocity'].type() != 'torch.cuda.FloatTensor':
                    state['velocity'] = state['velocity'].float()
                if 'master_param' in state.keys() and state['master_param'].type() != 'torch.cuda.FloatTensor':
                    state['master_param'] = state['master_param'].float()

                # Append all params, gradients, decays, velocity, momentum and updates to a list
                if p.type() == 'torch.cuda.HalfTensor':
                    fp16_grad_list.append(grad)
                    fp32_param_list.append(state['master_param'])
                    fp16_from_fp32_param_list.append(p.data)
                    fp16_per_param_decay.append(group['weight_decay'])
                    fp16_momentum.append(state["momentum"])
                    fp16_velocity.append(state["velocity"])
                    if not self.updates_created:
                        # self.fp16_update.append(torch.empty_like(p.data, dtype=torch.float32))
                        # Use fp16 weights as temporary buffer for update term.
                        # This is safe because fp16 weights are overwritten after apply_gradients
                        self.fp16_update.append(p.data)
                else:
                    grad_list.append(grad)
                    param_list.append(p.data)
                    per_param_decay.append(group['weight_decay'])
                    momentum.append(state["momentum"])
                    velocity.append(state["velocity"])
                    if not self.updates_created:
                        self.update.append(torch.empty_like(p.data))
                state['step'] += 1
        self.updates_created = True
        update = self.update
        fp16_update = self.fp16_update

        self.step_count = state['step']
        # Calculate learning rate from input schedule
        # if self.max_steps != -1:
        schedule_fct = SCHEDULES[self.schedule]
        lr_scheduled = self.learning_rate * schedule_fct(self.step_count / self.max_steps, self.warmup)
        if torch.distributed.get_rank() == 0:
            print("Step {} LR {}".format(self.step_count, lr_scheduled))
        # else:
        #     lr_scheduled = self.learning_rate

        overflow_buf = torch.cuda.IntTensor([0])

        if len(grad_list) > 0:
            self.apply_gradients(overflow_buf, lr_scheduled, per_param_decay, grad_list, param_list, momentum, velocity,
                                 update)
        if len(fp16_grad_list) > 0:
            self.apply_gradients(overflow_buf, lr_scheduled, fp16_per_param_decay, fp16_grad_list, fp32_param_list,
                                 fp16_momentum, fp16_velocity, fp16_update)
            multi_tensor_applier(
                scale,
                overflow_buf,
                [fp32_param_list, fp16_from_fp32_param_list],
                1.)

        return loss
'''