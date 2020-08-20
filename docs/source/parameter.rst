**********
Parameters
**********

Overview
--------

To run HetSeq, almost all the parameters are passed through commond line processed by ``argparse``. Thoses parameters can be grouped into several clusters: 

        * ``Extendable Components Parameters``: including ``task``, ``optimizer``, and  ``lr_scheduler``;
        * ``Distributed Parameters``: to set up distributed training enviroments; 
        * ``Training Prameters``: other important parameters to control stop criteria, logging information, checkpoints and etc.

Here we are going to explain most parameters in details.

Extendable Components Parameters
--------------------------------

* **Task**: ``--task``: Application name, its corresponding class defines major parts of the application. Currently support ``bert`` and ``mnist``, can be extended to other models.        


	* ``--task bert``:
		Extra parameters for ``bert`` task:
			* ``--data``: Dataset directory or file to be loaded in the corresponding task.
			* ``--config_file``: Configuration file of ``BERT`` model, example can be found `here <https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/bert_config.json>`__
			* ``--dict``: PATH of BPE dictionary for BERT model. Typically it has ~30,000 tokens.
			* ``--max_pred_length``: max number of tokens in a sentence, ``512`` by default.
			* ``--num_file``: number of input files for training, used with ``--data`` to debug. ``0`` by default to use all the data files.

                        
	* ``--task mnist``:
		Extra parameters for ``mnist`` task:
			* ``--data``: Dataset directory or file to be loaded in the corresponding task, compatible with ``torchvision.datasets.MNIST(path, train=True, download=True)``.


* **Optimizer**: ``--optimizer``: Optimizer defined in HetSeq is based on ``torch.optim.Optimizer`` with extra gradient and learning rate manipulation function. Currently support ``adam`` and ``adadelta`` which can be extended to many other optimizers.

        * ``--optimizer adam``:
          	Extra parameters for ``adam`` optimizer: `Fixed Weight Decay Regularization in Adam  <https://arxiv.org/abs/1711.05101>`__.
                        * ``--adam-betas``: betas to control momentum and velocity. Default='(0.9, 0.999)'.
                        * ``--adam-eps``: epsilon for avoiding deviding by 0. Default=1e-8.
                        * ``--weight-decay``: weight decay. ``0`` by default.                 


        * ``--optimizer adadelta``:
          	Extra parameters for ``adadelta`` optimizer:            
                        * ``--adadelta_rho``: Default=0.9.
                        * ``--adadelta_eps``: epsilon for avoiding deviding by 0. Default=1e-6.
                        * ``--dadelta_weight_decay``: ``0`` by default.


* **Lr_scheduler**: ``--lr_scheduler``: Learning rate scheduler defined in HetSeq customized to consider stop criteria ``end-learning-rate``, ``total-num-update`` and ``warmup-updates``. Currently support ``Polynomial Decay Scheduler``.

        * ``--optimizer PolynomialDecayScheduler``:
                Extra parameters for ``PolynomialDecayScheduler``:  
                        * ``--force-anneal``: force annealing at specified epoch, by default not existed.
                        * ``--power``: decay power. ``1.0`` by default. 
                        * ``--warmup-updates``: warmup the learning rate linearly for the first N updates, ``0`` by default.
                        * ``--total-num-update``: total number of update steps until learning rate decay to ``--end-learning-rate``, ``10000`` by default.
                        * ``--end-learning-rate``: learning rate when traing stops. ``0`` by default.


Distributed Parameters
----------------------
Distrbuted parameters play a key role in HetSeq to set up the distrbuted training environments, it defines the number of nodes, number of GPUs, communication methods and etc. 

* ``--fast-stat-sync``: Enable fast sync of stats between nodes, this hardcodes to sync only some default stats from logging_output. 
* ``--device-id``: index of single GPU used in the training. ``0`` by default. 

``torch.nn.parallel.distributed import DistributedDataParallel`` related parameters, see `document <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel>`__ for more informaiton. Our implementation consider input and put tensors on the same device.

        * ``--bucket-cap-mb``: ``25`` by default
        * ``--find-unused-parameters``: ``False`` by default
       
``torch.distributed.init_process_group`` related parameters, control the main environment of distributed training. See :ref:`distribute` or `document <https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group>`__ for more information. 
        * ``--ddp-backend``: distributed data parallel backend, currently only support ``c10d`` with ``NCCL`` to communicate between GPUs. Default: 'c10d'.
        * ``--distributed-init-method``: initial methods to communicate between GPUs. Default ``None``.
        * ``--distributed-world-size``: total number of GPUs/processes in the distributed seeting. Defalut: ``max(1, torch.cuda.device_count())``.
        * ``--distributed-rank``:  rank of the current GPU, ``0`` by default. 


`
Training Parameters
-------------------

* ``--max-epoch``: maximum epoches allowd in the training. ``0`` by default. 

* ``--max-update``:  maximum number of updates allowd in the training. ``0`` by default. 

* ``--required-batch-size-multiple``: check the batch size is the multiple times of the given number.  ``1`` by default. 

* ``--update-freq``: update parameters every ``N_i`` batches, when in epoch i. ``1`` by default. 

* ``--max-tokens``:  maximum number of tokens of a batch ``0``, not assigned.
  
* ``--max-sentences``:  maximum number of sentences/images/instances of a batch (batch size), not assigned.

.. note::
        
        ``--max-tokens`` or ``--max-sentences`` must be assigned in the prameter settings.

* ``--train-subset``: string to store training subset, ``train`` by default.

* ``--num-workers``: number of threads used in the data loading process.

* ``--save-interval-updates``: save a checkpoint (and validate) every N updates, ``0`` by default. 

* ``--seed``: onlu seed in the training process to control all the possible random steps (e.g. in ``torch``, ``numpy`` and ``random``). ``19940802`` by default.

* ``--log-interval``: log progress every N batches (when progress bar is disabled), ``1`` by default. 

* ``--log-format``: log format to use, choices=['none', 'simple'], ``simple`` by default.
