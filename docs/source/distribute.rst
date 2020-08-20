*******************
Distributed Setting
*******************


HetSeq can be executed on single GPU on a single node, multiple GPUs on a single node, or multiple GPUs across multiple nodes. Main logic is defined at `train.py <https://github.com/yifding/hetseq/blob/master/train.py#L213>`__.

Control Parameters
------------------
``--distributed-init-method``: defines an initialization. 

	e.g\: ``tcp://10.32.82.207:11111`` (IP address:port. TCP for multiple nodes) or 
	     ``file:///hetseq/communicate.txt`` (shared file for multiple nodes).

``--distributed-world-size``: total number of GPUs used in the training.

``--distributed-gpus``: the number of GPUs on the current node.

``--distributed-rank``: represents the rank/index of the first GPU used on current node.


Different Distributed Settings
------------------------------

1.Single GPU:

.. code:: console

	$ --distributed-world-size 1 --device-id 1

2. Four GPUs on a single node:

.. code:: console
	
	$ --distributed-world-size 4

3. Four nodes with four GPUs each (16 GPUs in total) ``10.00.123.456`` is the IP address of first node and ``11111`` is the port number:

* 1st node 

.. code:: console

	$ --distributed-init-method tcp://10.00.123.456:11111 --distributed-world-size 16 --distributed-gpus 4 --distributed-rank 0

* 2nd node 

.. code:: console

	$ --distributed-init-method tcp://10.00.123.456:11111 --distributed-world-size 16 --distributed-gpus 4 --distributed-rank 4

* 3rd node 

.. code:: console

	$ --distributed-init-method tcp://10.00.123.456:11111 --distributed-world-size 16 --distributed-gpus 4 --distributed-rank 8

* 4th node 

.. code:: console

	$ --distributed-init-method tcp://10.00.123.456:11111 --distributed-world-size 16 --distributed-gpus 4 --distributed-rank 12






Main Logic
----------

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





