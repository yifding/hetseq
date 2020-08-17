# HetSeq: Distributed GPU Training on Heterogeneous Infrastructure
This is our coding implementation for the paper:

>Yifan Ding, Nicholas Botzer, Tim Weninger (2020). 
HetSeq: Distributed GPU Training on Heterogeneous Infrastructure, to appear.

Author: Yifan Ding (yding4@nd.edu)

## Preparation for the reporsitory
1) create and activate conda virtual environment with Python 3.7.4 (recommended)
```
$ conda create --name hetseq
$ conda activate hetseq
$ conda install python=3.7.4
```

2) Git clone directory and install nessasory package
```
$ git clone https://github.com/yifding/hetseq.git
$ cd /path/to/hetseq
$ pip install -r requirement.txt 
$ pip install --editable . 
```

3) **To Run BERT:** Download data files including training corpus, model configuration, and BPE dictionary. Test corpus from [here](https://drive.google.com/file/d/1ZPJVAiV7PsewChi7xKACrjuniJ2N9Sry/view?usp=sharing), full data from [this link](https://drive.google.com/file/d/1Vq_UO-T9345uYs8a7zloukGfhDXSDd2A/view?usp=sharing). Download test_DATA.zip for test or DATA.zip for full run, unzip it and place the ```preprocessing/``` directory inside the package directory

## Distributed Configuration
HetSeq can be executed on single GPU on a single node, multiple GPUs on a single node, or multiple GPUs across multiple nodes. Main logic is defined at [train.py](https://github.com/yifding/hetseq/blob/master/train.py#L213)

* **--distributed-init-method**: defines an initialization. e.g.: "tcp://10.32.82.207:11111" (tcp for multiple nodes) or "file:///hetseq/communicate.txt" (shared file for multiple nodes).
* **--distributed-world-size**: total number of GPUs used in the training.
* **--distributed-gpus**: the number of GPUs on the current node.
* **--distributed-rank**: represents the rank/index of the first GPU used on current node. 

### Set up different distributed settings:
#### 1. single GPU: ```--distributed-world-size 1 --device-id 1```
#### 2. Four GPUs on a single node: ```--distributed-world-size 4```
#### 3. Four nodes with four GPUs each (16 GPUs in total)  "10.00.123.456" is the IP address of first node and "11111" is the port number:
##### 1st node: ```--distributed-init-method tcp://10.00.123.456:11111 --distributed-world-size 16 --distributed-gpus 4 --distributed-rank 0```
##### 2nd node: ```--distributed-init-method tcp://10.00.123.456:11111 --distributed-world-size 16 --distributed-gpus 4 --distributed-rank 4```
##### 3rd node: ```--distributed-init-method tcp://10.00.123.456:11111 --distributed-world-size 16 --distributed-gpus 4 --distributed-rank 8```
##### 4th node: ```--distributed-init-method tcp://10.00.123.456:11111 --distributed-world-size 16 --distributed-gpus 4 --distributed-rank 12```

## Example to Run the Codes
Set the directory path to $DIST. 
* For example, if the directory path is under the default path.
```
$ DIST=~/hetseq
```

### Single GPU
```
python3 ${DIST}/train.py  \
--task bert   --data ${DIST}/preprocessing/test_128/ \
--dict ${DIST}/preprocessing/uncased_L-12_H-768_A-12/vocab.txt  \
--config_file ${DIST}/preprocessing/uncased_L-12_H-768_A-12/bert_config.json  \
--max-sentences 32  --fast-stat-sync --max-update 900000 --update-freq 4  \
--valid-subset test --num-workers 4 \
--warmup-updates 10000  --total-num-update 1000000 --lr 0.0001  \
--weight-decay 0.01 --distributed-world-size 1  \
--device-id 1 --save-dir sing_gpu
```
### Multiple GPU on a single Node, examples are all four GPUs on one Node. 
```
python3 ${DIST}/train.py  \
--task bert   --data ${DIST}/preprocessing/test_128/ \
--dict ${DIST}/preprocessing/uncased_L-12_H-768_A-12/vocab.txt  \
--config_file ${DIST}/preprocessing/uncased_L-12_H-768_A-12/bert_config.json  \
--max-sentences 32  --fast-stat-sync --max-update 900000 --update-freq 4  \
--valid-subset test --num-workers 4 \
--warmup-updates 10000  --total-num-update 1000000 --lr 0.0001  \
--weight-decay 0.01 \
--save-dir node1gpu4
```

### Multiple GPU on multiple Nodes, examples are four nodes with four GPUs each.
1) set the IP address and port, (a fake example is given)
```
$AD=10.00.123.456:11111
```
2) on the main node, run:
```
python3 ${DIST}/train.py  \
--task bert   --data ${DIST}/preprocessing/test_128/ \
--dict ${DIST}/preprocessing/uncased_L-12_H-768_A-12/vocab.txt  \
--config_file ${DIST}/preprocessing/uncased_L-12_H-768_A-12/bert_config.json  \
--max-sentences 32  --fast-stat-sync --max-update 900000 --update-freq 4  \
--valid-subset test --num-workers 4 \
--warmup-updates 10000  --total-num-update 1000000 --lr 0.0001  \
--weight-decay 0.01 --save-dir node4gpu4  \
--distributed-init-method tcp://${AD} \
--distributed-world-size 16 \
--distributed-gpus 4 \
--distributed-rank 0
```
3) on the other 3 nodes, run the same code except --distributed-rank and saving file name. 
* 3-1) second node:
```
python3 ${DIST}/train.py  \
--task bert   --data ${DIST}/preprocessing/test_128/ \
--dict ${DIST}/preprocessing/uncased_L-12_H-768_A-12/vocab.txt  \
--config_file ${DIST}/preprocessing/uncased_L-12_H-768_A-12/bert_config.json  \
--max-sentences 32  --fast-stat-sync --max-update 900000 --update-freq 4  \
--valid-subset test --num-workers 4 \
--warmup-updates 10000  --total-num-update 1000000 --lr 0.0001  \
--weight-decay 0.01 --save-dir node4gpu4  \
--distributed-init-method tcp://${AD} \
--distributed-world-size 16 \
--distributed-gpus 4 \
--distributed-rank 4 
```

* 3-2) third node:
```
python3 ${DIST}/train.py  \
--task bert   --data ${DIST}/preprocessing/test_128/ \
--dict ${DIST}/preprocessing/uncased_L-12_H-768_A-12/vocab.txt  \
--config_file ${DIST}/preprocessing/uncased_L-12_H-768_A-12/bert_config.json  \
--max-sentences 32  --fast-stat-sync --max-update 900000 --update-freq 4  \
--valid-subset test --num-workers 4 \
--warmup-updates 10000  --total-num-update 1000000 --lr 0.0001  \
--weight-decay 0.01 --save-dir node4gpu4  \
--distributed-init-method tcp://${AD} \
--distributed-world-size 16 \
--distributed-gpus 4 \
--distributed-rank 8
```

* 3-3) fourth node:
```
python3 ${DIST}/train.py  \
--task bert   --data ${DIST}/preprocessing/test_128/ \
--dict ${DIST}/preprocessing/uncased_L-12_H-768_A-12/vocab.txt  \
--config_file ${DIST}/preprocessing/uncased_L-12_H-768_A-12/bert_config.json  \
--max-sentences 32  --fast-stat-sync --max-update 900000 --update-freq 4  \
--valid-subset test --num-workers 4 \
--warmup-updates 10000  --total-num-update 1000000 --lr 0.0001  \
--weight-decay 0.01 --save-dir node4gpu4  \
--distributed-init-method tcp://${AD} \
--distributed-world-size 16 \
--distributed-gpus 4 \
--distributed-rank 12
```


## Available corpus under ```preprocessing/```, 
* phase one of BERT training corpus : ```preprocessing/hdf5_lower_case_1_seq_len_128.../wikicorpus_en/```
* phase two of BERT training corpus : ```preprocessing/hdf5_lower_case_1_seq_len_512.../wikicorpus_en/```
* sample test for phase one : ```preprocessing/test128/```
* sample test for phase two : ```preprocessing/test512/```
* see [NVIDIA-pytorch-BERT](https://arxiv.org/abs/1810.04805), [google_original_BERT](https://github.com/google-research/bert) and [BERT paper](https://arxiv.org/abs/1810.04805) for more information.
* current provided is generated from [NVIDIA-pytorch-BERT](https://arxiv.org/abs/1810.04805) with wikipedia data (book data is not available)


## Available BERT model configuration under ```preprocessing/```, 
* BERT-base: preprocessing/uncased_L-12_H-768_A-12
* BERT-large: preprocessing/uncased_L-24_H-1024_A-16

## Components
### Portable 
* tasks: BERT (default) ```tasks.py```
* optimizer: Adam (default) ```optim.py```
* learning rate scheduler: Polynomial Decay Scheduler (default) ```lr_scheduler.py```
* dataset: BERT h5py dataset (default) ```data/h5pyDataset.py```

### Core
* BERT model: ```bert_modeling.py```
* main train code: ```train.py```
* distrbuted training code: ```controller.py```

### Supporting
* utils: ```utils.py```
* file utils: ```file_utils.py```
* checkpoints utils: ```checkpoint_utils.py```
* distributed utils: ```distributed_utils.py``` (IMPORTANT)


## Parameter explanation (TODO)
```
usage:  [-h] [--no-progress-bar] [--seed N] [--cpu] [--log-interval N][--log-format {none,simple}] 
        [--num-workers N] [--max-tokens N] [--max-sentences N] [--required-batch-size-multiple N] 
        [--train-subset SPLIT] [--valid-subset SPLIT] [--validate-interval N] [--disable-validation] 
        [--max-tokens-valid N] [--max-sentences-valid N] [--curriculum N] [--task TASK] [--data DATA] 
        [--dict PATH of a file] [--config_file PATH of a file] [--max_pred_length MAX_PRED_LENGTH] 
        [--num_file NUM_FILE] [--distributed-world-size N] [--distributed-rank DISTRIBUTED_RANK] 
        [--distributed-gpus DISTRIBUTED_GPUS] [--distributed-backend DISTRIBUTED_BACKEND] 
        [--distributed-init-method DISTRIBUTED_INIT_METHOD] [--device-id DEVICE_ID] 
        [--distributed-no-spawn] [--ddp-backend {c10d}] [--bucket-cap-mb MB] [--fix-batches-to-gpus] 
        [--find-unused-parameters] [--fast-stat-sync] [--max-epoch N] [--max-update N] 
        [--clip-norm NORM] [--update-freq N1,N2,...,N_K] [--lr LR_1,LR_2,...,LR_N] [--min-lr LR] 
        [--use-bmuf] [--optimizer OPTIMIZER] [--adam-betas B] [--adam-eps D] [--weight-decay WD] 
        [--lr_scheduler LR_SCHEDULER] [--force-anneal N] [--warmup-updates N] 
        [--end-learning-rate END_LEARNING_RATE] [--power POWER] [--total-num-update TOTAL_NUM_UPDATE] 
        [--save-dir DIR] [--restore-file RESTORE_FILE] [--reset-dataloader] [--reset-lr-scheduler] 
        [--reset-meters] [--reset-optimizer] [--optimizer-overrides DICT] [--save-interval N]
        [--save-interval-updates N] [--keep-interval-updates N] [--keep-last-epochs N] [--no-save] 
        [--no-epoch-checkpoints] [--no-last-checkpoints] [--no-save-optimizer-state] 
        [--best-checkpoint-metric BEST_CHECKPOINT_METRIC] [--maximize-best-checkpoint-metric]
```


## Performance table
| nodes | GPUs | epochs | batch size |  steps  | avg. time per step | training time | training loss | expansion | speedup |
|:-----:|:----:|:------:|:----------:|:-------:|:------------------:|:-------------:|:-------------:|:---------:|:-------:|
|   1   |   4  |    5   |     128    | 267,139 |        2.60s       |     7.19d     |     0.026     |     1     |    1    |
|   2   |   8  |    5   |     256    | 133,570 |        2.69s       |     4.19d     |     0.028     |    0.86   |   1.72  |
|   4   |  16  |    5   |     512    |  66,785 |        2.794       |     2.23d     |     0.031     |    0.81   |   3.22  |
|   8   |  32  |    5   |     1024   |  33,393 |        3.126       |     1.21d     |     0.055     |    0.74   |   5.94  |
## Notice and tips
> loading data may take a while. 

## License
this repository is MIT-licensed. It is created based on [fairseq](https://github.com/pytorch/fairseq), 
[NVIDIA-BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT), and 
[pytorch](https://github.com/pytorch/pytorch) 




