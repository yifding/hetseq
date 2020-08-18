********
Examples
********


BERT Task
---------


1. Single GPU

.. code-block:: console

	python3 ${DIST}/train.py  \
		--task bert   --data ${DIST}/preprocessing/test_128/ \
		--dict ${DIST}/preprocessing/uncased_L-12_H-768_A-12/vocab.txt  \
		--config_file ${DIST}/preprocessing/uncased_L-12_H-768_A-12/bert_config.json  \
		--max-sentences 32  --fast-stat-sync --max-update 900000 --update-freq 4  \
		--valid-subset test --num-workers 4 \
		--warmup-updates 10000  --total-num-update 1000000 --lr 0.0001  \
		--weight-decay 0.01 --distributed-world-size 1  \
		--device-id 1 --save-dir sing_gpu

2. Multiple GPU on a single Node, examples are all four GPUs on one Node.

.. code:: console

	python3 ${DIST}/train.py  \
		--task bert   --data ${DIST}/preprocessing/test_128/ \
		--dict ${DIST}/preprocessing/uncased_L-12_H-768_A-12/vocab.txt  \
		--config_file ${DIST}/preprocessing/uncased_L-12_H-768_A-12/bert_config.json  \
		--max-sentences 32  --fast-stat-sync --max-update 900000 --update-freq 4  \
		--valid-subset test --num-workers 4 \
		--warmup-updates 10000  --total-num-update 1000000 --lr 0.0001  \
		--weight-decay 0.01 \
		--save-dir node1gpu4

3. Multiple GPUs on multiple nodes, examples are two nodes with four GPUs each.


* on the main node

.. code:: console

	python3 ${DIST}/train.py  \
		--task bert   --data ${DIST}/preprocessing/test_128/ \
		--dict ${DIST}/preprocessing/uncased_L-12_H-768_A-12/vocab.txt  \
		--config_file ${DIST}/preprocessing/uncased_L-12_H-768_A-12/bert_config.json  \
		--max-sentences 32  --fast-stat-sync --max-update 900000 --update-freq 4  \
		--valid-subset test --num-workers 4 \
		--warmup-updates 10000  --total-num-update 1000000 --lr 0.0001  \
		--weight-decay 0.01 --save-dir node4gpu4  \
		--distributed-init-method tcp://10.00.123.456:11111 \
		--distributed-world-size 8 --distributed-gpus 4 --distributed-rank 0

* on the other second node

.. code:: console

	python3 ${DIST}/train.py  \
		--task bert   --data ${DIST}/preprocessing/test_128/ \
		--dict ${DIST}/preprocessing/uncased_L-12_H-768_A-12/vocab.txt  \
		--config_file ${DIST}/preprocessing/uncased_L-12_H-768_A-12/bert_config.json  \
		--max-sentences 32  --fast-stat-sync --max-update 900000 --update-freq 4  \
		--valid-subset test --num-workers 4 \
		--warmup-updates 10000  --total-num-update 1000000 --lr 0.0001  \
		--weight-decay 0.01 \
		--distributed-init-method tcp://10.00.123.456:11111 \
		--distributed-world-size 8 --distributed-gpus 4 --distributed-rank 4


