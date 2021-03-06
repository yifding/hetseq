#!/bin/bash

#$-m abe
#$-M yding4@nd.edu
#$-q gpu@qa-p100-005 # specify the queue
#$-l gpu_card=4
#$-N node4gpu16_mnist_sub2

export PATH=/afs/crc.nd.edu/user/y/yding4/Transformer/bin:$PATH
export LD_LIBRARY_PATH=/afs/crc.nd.edu/user/y/yding4/Transformer/lib:$LD_LIBRARY_PATH

DIST=/scratch365/yding4/hetseq
AD=tcp://10.32.82.207:11111

python3 ${DIST}/train.py  \
    --task mnist  --optimizer adadelta --lr-scheduler PolynomialDecayScheduler  \
    --data /scratch365/yding4/mnist/MNIST/processed/  --clip-norm 100 \
    --max-sentences 64  --fast-stat-sync --max-epoch 20 --update-freq 1  \
    --valid-subset test --num-workers 4 \
    --warmup-updates 0  --total-num-update 50000 --lr 1.01  \
    --distributed-init-method ${AD}  --distributed-world-size 16 \
    --distributed-gpus 4 --distributed-rank 8 --save-dir node4gpu16