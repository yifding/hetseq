#!/bin/bash

#$-m abe
#$-M yding4@nd.edu
#$-q gpu@qa-xp-002 # specify the queue
#$-l gpu_card=4
#$-N node1gpu4_mnist_main

#conda environment is on ~/Transformer
export PATH=/afs/crc.nd.edu/user/y/yding4/.conda/envs/hetseq/bin:$PATH
export LD_LIBRARY_PATH=/afs/crc.nd.edu/user/y/yding4/.conda/envs/hetseq/lib:$LD_LIBRARY_PATH

DIST=/scratch365/yding4/hetseq

python3 ${DIST}/hetseq/train.py  \
    --task mnist  --optimizer adadelta --lr-scheduler PolynomialDecayScheduler  \
    --data /scratch365/yding4/mnist  --clip-norm 100 \
    --max-sentences 64  --fast-stat-sync --max-epoch 20 --update-freq 1  \
    --valid-subset test --num-workers 4 \
    --warmup-updates 0  --total-num-update 50000 --lr 1.01  \
    --save-dir node1gpu4 2>&1 | tee node1gpu4.log