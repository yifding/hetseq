*******************
Distributed Setting
*******************

Overview
--------

HetSeq can be executed on single GPU on a single node, multiple GPUs on a single node, or multiple GPUs across multiple nodes. Main logic is defined at `train.py <https://github.com/yifding/hetseq/blob/master/train.py#L213>`__.

.. code:: python

	if args.distributed_init_method is not None:
       		assert args.distributed_gpus <= torch.cuda.device_count()

        	if args.distributed_gpus > 1 and not args.distributed_no_spawn:
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
        	main(args)


Configuration
-------------

.. autoclass:: tasks.LanguageModelingTask



