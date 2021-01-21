#!/bin/bash

#$-m abe
#$-M yding4@nd.edu
#$-q gpu@qa-xp-003 # specify the queue
#$-l gpu_card=4
#$-N eval_ner_fine_tuning

CODE=/home/yding4/EL_resource/baseline/deep_ed_PyTorch/deep_ed_PyTorch/entities/learn_e2v/YD_eval_entity_embedding.py

python3 ${CODE} \
    --dir /home/yding4/EL_resource/data/deep_ed_PyTorch_data/generated/ent_vecs \
    --root_data_dir /home/yding4/EL_resource/data/deep_ed_PyTorch_data/