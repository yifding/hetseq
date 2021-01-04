#!/bin/bash

#$-m abe
#$-M yding4@nd.edu
#$-q gpu@qa-xp-002 # specify the queue
#$-l gpu_card=4
#$-N node8gpu32_mnist_main

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
    --distributed-init-method ${AD}  --distributed-world-size 32 \
    --distributed-gpus 4 --distributed-rank 0 --save-dir node8gpu32 2>&1 | tee node8gpu32.log