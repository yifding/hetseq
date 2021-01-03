#!/bin/bash

#$-m abe
#$-M yding4@nd.edu
#$-q gpu@qa-xp-003 # specify the queue
#$-l gpu_card=4
#$-N eval_ner_fine_tuning

export PATH=/afs/crc.nd.edu/user/y/yding4/.conda/envs/hetseq/bin:$PATH
export LD_LIBRARY_PATH=/afs/crc.nd.edu/user/y/yding4/.conda/envs/hetseq/lib:$LD_LIBRARY_PATH

DIST=/scratch365/yding4/hetseq
CODE=/scratch365/yding4/hetseq/test/test_eval_bert_fine_tuning.py
EXTENSION_FILE=/scratch365/yding4/EL_resource/preprocess/build_datasets_huggingface/ace2004.py

HETSEQ_STATE_DICT=/scratch365/yding4/hetseq/CRC_RUN_FILE/run_hetseq_bert_tine_tuning_ner/bert_fine_tuning_ner/checkpoint_last.pt
DATA_DIR=/scratch365/yding4/EL_resource/data/processed/EL_CONLL_NER_stanza/
TEST_FILE=ace2004.conll


python3 ${CODE} \
    --test_file   ${DATA_DIR}${TEST_FILE} \
    --extension_file    ${EXTENSION_FILE} \
    --config_file   ${DIST}/preprocessing/uncased_L-12_H-768_A-12/bert_config.json  \
    --vocab_file    ${DIST}/preprocessing/uncased_L-12_H-768_A-12/vocab.txt  \
    --hetseq_state_dict ${HETSEQ_STATE_DICT}