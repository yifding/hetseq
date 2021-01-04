#!/bin/bash

#$-m abe
#$-M yding4@nd.edu
#$-q gpu@qa-xp-008 # specify the queue
#$-l gpu_card=4
#$-N bert_sub1

#conda environment is on ~/Transformer
export PATH=/afs/crc.nd.edu/user/y/yding4/.conda/envs/hetseq/bin:$PATH
export LD_LIBRARY_PATH=/afs/crc.nd.edu/user/y/yding4/.conda/envs/hetseq/lib:$LD_LIBRARY_PATH

DIST=/scratch365/yding4/hetseq

python3 ${DIST}/hetseq/train.py  \
--task bert   --data ${DIST}/preprocessing/test_128/ \
--dict ${DIST}/preprocessing/uncased_L-12_H-768_A-12/vocab.txt  \
--config_file ${DIST}/preprocessing/uncased_L-12_H-768_A-12/bert_config.json     \
--max-sentences 32  --fast-stat-sync --max-update 450000 --update-freq 4 \
--valid-subset test --num-workers 4 \
--warmup-updates 10000  --total-num-update 1000000 --lr 0.0001 \
--weight-decay 0.01 \
--save-dir bsz_512-num_sentece_32_gpu_8-update_4-phase_1 \
--distributed-init-method tcp://10.32.82.211:12345 \
--distributed-world-size 8 \
--distributed-gpus 4 \
--distributed-rank 4 \
2>&1 | tee bert_bsz_512-num_sentece_32_gpu_8-update_4-phase_1_sub.log
# bsz_4-gpu_4-update_16-phase_1-file_cent.log
