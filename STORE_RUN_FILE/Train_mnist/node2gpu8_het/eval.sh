#!/bin/bash

#$-m abe
#$-M yding4@nd.edu
#$-q gpu@qa-xp-002 # specify the queue
#$-l gpu_card=4
#$-N node2gpu8_mnist_main1

export PATH=/afs/crc.nd.edu/user/y/yding4/Transformer/bin:$PATH
export LD_LIBRARY_PATH=/afs/crc.nd.edu/user/y/yding4/Transformer/lib:$LD_LIBRARY_PATH

code=/scratch365/yding4/hetseq/eval_mnist.py
ckpt=/scratch365/yding4/hetseq/CRC_RUN_FILE/Train_mnist/node2gpu8_het/node2gpu8/checkpoint_last.pt
mnist=/scratch365/yding4/mnist/MNIST/processed

python3 ${code} --model_ckpt ${ckpt} --mnist_dir ${mnist} 2>&1 |tee eval.log