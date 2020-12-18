**************
Running Script
**************


BERT Task
---------


1. Single GPU

.. code-block:: console

	$ DIST=~/hetseq
	$ python3 ${DIST}/hetseq/train.py  \
		--task bert   --data ${DIST}/preprocessing/test_128/ \
		--dict ${DIST}/preprocessing/uncased_L-12_H-768_A-12/vocab.txt  \
		--config_file ${DIST}/preprocessing/uncased_L-12_H-768_A-12/bert_config.json  \
		--max-sentences 32  --fast-stat-sync --max-update 900000 --update-freq 4  \
		--valid-subset test --num-workers 4 \
		--warmup-updates 10000  --total-num-update 1000000 --lr 0.0001  \
		--weight-decay 0.01 --distributed-world-size 1  \
		--device-id 0 --save-dir bert_single_gpu

2. Multiple GPU on a single Node, examples are all four GPUs on one Node.

.. code:: console

	$ DIST=~/hetseq
	$ python3 ${DIST}/hetseq/train.py  \
		--task bert   --data ${DIST}/preprocessing/test_128/ \
		--dict ${DIST}/preprocessing/uncased_L-12_H-768_A-12/vocab.txt  \
		--config_file ${DIST}/preprocessing/uncased_L-12_H-768_A-12/bert_config.json  \
		--max-sentences 32  --fast-stat-sync --max-update 900000 --update-freq 4  \
		--valid-subset test --num-workers 4 \
		--warmup-updates 10000  --total-num-update 1000000 --lr 0.0001  \
		--weight-decay 0.01 \
		--save-dir bert_node1gpu4

3. Multiple GPUs on multiple nodes, examples are two nodes with four GPUs each.


* on the main node

.. code:: console

	$ DIST=~/hetseq
	$ python3 ${DIST}/hetseq/train.py  \
		--task bert   --data ${DIST}/preprocessing/test_128/ \
		--dict ${DIST}/preprocessing/uncased_L-12_H-768_A-12/vocab.txt  \
		--config_file ${DIST}/preprocessing/uncased_L-12_H-768_A-12/bert_config.json  \
		--max-sentences 32  --fast-stat-sync --max-update 900000 --update-freq 4  \
		--valid-subset test --num-workers 4 \
		--warmup-updates 10000  --total-num-update 1000000 --lr 0.0001  \
		--weight-decay 0.01 --save-dir bert_node2gpu4  \
		--distributed-init-method tcp://10.00.123.456:11111 \
		--distributed-world-size 8 --distributed-gpus 4 --distributed-rank 0

* on the other second node

.. code:: console

	$ DIST=~/hetseq
	$ python3 ${DIST}/hetseq/train.py  \
		--task bert   --data ${DIST}/preprocessing/test_128/ \
		--dict ${DIST}/preprocessing/uncased_L-12_H-768_A-12/vocab.txt  \
		--config_file ${DIST}/preprocessing/uncased_L-12_H-768_A-12/bert_config.json  \
		--max-sentences 32  --fast-stat-sync --max-update 900000 --update-freq 4  \
		--valid-subset test --num-workers 4 \
		--warmup-updates 10000  --total-num-update 1000000 --lr 0.0001  \
		--weight-decay 0.01 \
		--distributed-init-method tcp://10.00.123.456:11111 \
		--distributed-world-size 8 --distributed-gpus 4 --distributed-rank 4



MNIST Task
----------

1. Single GPU

.. code:: console

	$ DIST=~/hetseq
	$ python3 ${DIST}/hetseq/train.py  \
    		--task mnist  --optimizer adadelta --lr-scheduler PolynomialDecayScheduler  \
    		--data ${DIST}  --clip-norm 100 \
    		--max-sentences 64  --fast-stat-sync --max-epoch 20 --update-freq 1  \
    		--valid-subset test --num-workers 4 \
    		--warmup-updates 0  --total-num-update 50000 --lr 1.01  \
    		--distributed-world-size 1 --device-id 0 --save-dir mnist_single_node


2. Multiple GPU on a single Node, examples are all four GPUs on one Node.

.. code:: console

	$ DIST=~/hetseq
	$ python3 ${DIST}/hetseq/train.py  \
    		--task mnist  --optimizer adadelta --lr-scheduler PolynomialDecayScheduler  \
    		--data ${DIST}  --clip-norm 100 \
    		--max-sentences 64  --fast-stat-sync --max-epoch 20 --update-freq 1  \
    		--valid-subset test --num-workers 4 \
    		--warmup-updates 0  --total-num-update 50000 --lr 1.01  \
    		--save-dir mnist_node1gpu4

3. Multiple GPUs on multiple nodes, examples are two nodes with four GPUs each.


* on the main node

.. code:: console
	
	$ DIST=~/hetseq
	$ python3 ${DIST}/hetseq/train.py  \
    		--task mnist  --optimizer adadelta --lr-scheduler PolynomialDecayScheduler  \
    		--data ${DIST}  --clip-norm 100 \
    		--max-sentences 64  --fast-stat-sync --max-epoch 20 --update-freq 1  \
    		--valid-subset test --num-workers 4 \
    		--warmup-updates 0  --total-num-update 50000 --lr 1.01  \
    		--save-dir mnist_node2gpu4 \
		--distributed-init-method tcp://10.00.123.456:11111 \
		--distributed-world-size 8 --distributed-gpus 4 --distributed-rank 0

* on the other second node

.. code:: console

	$ DIST=~/hetseq
	$ python3 ${DIST}/hetseq/train.py  \
    		--task mnist  --optimizer adadelta --lr-scheduler PolynomialDecayScheduler  \
    		--data ${DIST}  --clip-norm 100 \
    		--max-sentences 64  --fast-stat-sync --max-epoch 20 --update-freq 1  \
    		--valid-subset test --num-workers 4 \
    		--warmup-updates 0  --total-num-update 50000 --lr 1.01  \
		--distributed-init-method tcp://10.00.123.456:11111 \
		--distributed-world-size 8 --distributed-gpus 4 --distributed-rank 4


Evaluate MNIST Task
-------------------

.. code:: console

	$ DIST=~/hetseq
	$ python3 ${DIST}/hetseq/eval_mnist.py --model_ckpt /path/to/check/point --mnist_dir ${DIST}
