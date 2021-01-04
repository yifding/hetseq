#!/bin/bash

#$-m abe
#$-M yding4@nd.edu
#$-q gpu@qa-p100-007 # specify the queue
#$-l gpu_card=1
#$-N run_bert_fine_tuning_ner

export PATH=/afs/crc.nd.edu/user/y/yding4/.conda/envs/hetseq/bin:$PATH
export LD_LIBRARY_PATH=/afs/crc.nd.edu/user/y/yding4/.conda/envs/hetseq/lib:$LD_LIBRARY_PATH

EXTENSION_FILE=/scratch365/yding4/EL_resource/preprocess/build_datasets_huggingface/BI_local_conll2003.py
DATA_DIR=/scratch365/yding4/EL_resource/data/CONLL2003/
TRAIN_FILE=train.txt
VALIDATION_FILE=valid.txt
TEST_FILE=test.txt
HETSEQ_STATE_DICT=/scratch365/yding4/Reddit_EL/RUN_FILE/Reddit_1st_test/node2gpu4/Reddit_512-num_sentece_6_gpu_8-update_4-phase_1/checkpoint3.pt

DIST=/scratch365/yding4/hetseq

python3 ${DIST}/hetseq/train.py  \
    --task BertForTokenClassification  --optimizer adam --lr-scheduler PolynomialDecayScheduler  \
    --fast-stat-sync --max-update 5000 --update-freq 1  \
    --valid-subset test --num-workers 4 \
    --warmup-updates 0  --total-num-update 50000 --lr 0.0001  \
    --dict ${DIST}/preprocessing/uncased_L-12_H-768_A-12/vocab.txt  \
    --config_file ${DIST}/preprocessing/uncased_L-12_H-768_A-12/bert_config.json    \
    --hetseq_state_dict ${HETSEQ_STATE_DICT}    \
    --train_file     ${DATA_DIR}${TRAIN_FILE}   \
    --validation_file     ${DATA_DIR}${VALIDATION_FILE} \
    --test_file   ${DATA_DIR}${TEST_FILE}   \
    --extension_file  ${EXTENSION_FILE} \
    --max-sentences 32 \
    --num-workers 8 \
    --load_state_dict_strict "False"    \
    --find-unused-parameters    \
    --save-dir bert_fine_tuning_ner
    # --distributed-world-size 1  --device-id 1 \


