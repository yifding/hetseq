#!/bin/bash

#$-m abe
#$-M yding4@nd.edu
#$-q gpu@qa-xp-003 # specify the queue
#$-l gpu_card=4
#$-N run_ner

export PATH=/afs/crc.nd.edu/user/y/yding4/.conda/envs/hetseq/bin:$PATH
export LD_LIBRARY_PATH=/afs/crc.nd.edu/user/y/yding4/.conda/envs/hetseq/lib:$LD_LIBRARY_PATH

DIST=/home/yding4/hetseq
CODE=/home/yding4/hetseq/test/test_conll_2003_dataset.py
EXTENSION_FILE=/home/yding4/EL_resource/preprocess/build_datasets_huggingface/BI_local_conll2003.py

TRANSFORMERS_STATE_DICT=/home/yding4/hetseq/preprocessing/transformers_ckpt/pytorch_model.bin
DATA_DIR=/home/yding4/EL_resource/data/CONLL2003/
TRAIN_FILE=train.txt
VALIDATION_FILE=valid.txt
TEST_FILE=test.txt



python3 ${CODE} \
    --train_file     ${DATA_DIR}${TRAIN_FILE} \
    --validation_file     ${DATA_DIR}${VALIDATION_FILE} \
    --test_file   ${DATA_DIR}${TEST_FILE} \
    --extension_file    ${EXTENSION_FILE} \
    --config_file   ${DIST}/preprocessing/uncased_L-12_H-768_A-12/bert_config.json  \
    --vocab_file    ${DIST}/preprocessing/uncased_L-12_H-768_A-12/vocab.txt  \
    --transformers_state_dict ${TRANSFORMERS_STATE_DICT}