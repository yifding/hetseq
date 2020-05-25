# BERT at a University: Distributed GPU training on Heterogeneous Systems
This is our coding implementation for the paper:

>Yifan Ding, Nicholas Botzer, Tim Weninger (2020). 
BERT at a University: Distributed GPU training on Heterogeneous Systems, to appear.

Author: Yifan Ding (yding4@nd.edu)

## Environment Requirement
The code has been tested running under Python 3.7.4. The required packages are as follows (same as requirement.txt):
* cython == 0.29.13
* numpy == 1.17.0
* h5py == 2.10.0
* torch == 1.2.0
* tqdm == 4.36.1
* boto3 == 1.9.244
* chardet == 3.0.4
* idna == 2.8
* python-dateutil == 2.8.0

## Preparation for the reporsitory
1) Install packages with conda virtual environment wity Python 3.7.4 (recommended)
```
$ conda create --name distribute_bert
$ conda activate distribute_bert
$ conda install python=3.7.4
```

2) Git clone directory and install nessasory package
```
$ git clone url (TODO)
$ cd url_directory (TODO)
$ pip install -r requirement.txt 
$ pip install --editable . 
```

3) Download data files including training corpus, model configuration, and BPE dictionary. From [this link](https://drive.google.com/file/d/). Download DATA.zip, unzip it and place the preprocessing directory inside the package directory


## Example to Run the Codes
Set the directory path to $DIST. 
* For example, if the directory path is under the default path.
```
$ DIST=~/Distributed_BERT/
```

### Single GPU
```
python3 ${DIST}/train.py  \
--task bert   --data {DIST}/preprocessing/test_128/ \
--dict {DIST}/preprocessing//uncased_L-12_H-768_A-12/vocab.txt  \
--config_file {DIST}/preprocessing//uncased_L-12_H-768_A-12/bert_config.json  \
--max-sentences 32  --fast-stat-sync --max-update 900000 --update-freq 4  \
--valid-subset test --num-workers 4 \
--warmup-updates 10000  --total-num-update 1000000 --lr 0.0001  \
--weight-decay 0.01 --distributed-world-size 1  \
--device-id 1 --save-dir sing_gpu_test  \
2>&1 | sing_gpu_test.log
```
### Multiple GPU on a single Node, examples are all four GPUs on one Node. 
```
python3 ${DIST}/train.py  \
--task bert   --data {DIST}/preprocessing/test_128/ \
--dict {DIST}/preprocessing//uncased_L-12_H-768_A-12/vocab.txt  \
--config_file {DIST}/preprocessing//uncased_L-12_H-768_A-12/bert_config.json  \
--max-sentences 32  --fast-stat-sync --max-update 900000 --update-freq 4  \
--valid-subset test --num-workers 4 \
--warmup-updates 10000  --total-num-update 1000000 --lr 0.0001  \
--weight-decay 0.01 \
--save-dir node1gpu4  2>&1 | node1gpu4.log
```

### Multiple GPU on multiple Nodes, examples are four nodes with four GPUs each.
1) set the IP address and port, (a fake example is given)
```
$AD=10.00.123.456:11111
```
2) on the main node, run:
```
python3 ${DIST}/train.py  \
--task bert   --data {DIST}/preprocessing/test_128/ \
--dict {DIST}/preprocessing//uncased_L-12_H-768_A-12/vocab.txt  \
--config_file {DIST}/preprocessing//uncased_L-12_H-768_A-12/bert_config.json  \
--max-sentences 32  --fast-stat-sync --max-update 900000 --update-freq 4  \
--valid-subset test --num-workers 4 \
--warmup-updates 10000  --total-num-update 1000000 --lr 0.0001  \
--weight-decay 0.01 --save-dir node1gpu4  \
--distributed-init-method tcp://{AD} \
--distributed-world-size 16 \
--distributed-gpus 4 \
--distributed-rank 0 \
2>&1 | node4gpu4_main.log
```
3) on the other 3 nodes, run the same code except --distributed-rank and saving file name. 
* 3-1) second node:
```
python3 ${DIST}/train.py  \
--task bert   --data {DIST}/preprocessing/test_128/ \
--dict {DIST}/preprocessing//uncased_L-12_H-768_A-12/vocab.txt  \
--config_file {DIST}/preprocessing//uncased_L-12_H-768_A-12/bert_config.json  \
--max-sentences 32  --fast-stat-sync --max-update 900000 --update-freq 4  \
--valid-subset test --num-workers 4 \
--warmup-updates 10000  --total-num-update 1000000 --lr 0.0001  \
--weight-decay 0.01 --save-dir node1gpu4  \
--distributed-init-method tcp://{AD} \
--distributed-world-size 16 \
--distributed-gpus 4 \
--distributed-rank 4 \
2>&1 | node4gpu4_sub1.log
```

* 3-2) third node:
```
python3 ${DIST}/train.py  \
--task bert   --data {DIST}/preprocessing/test_128/ \
--dict {DIST}/preprocessing//uncased_L-12_H-768_A-12/vocab.txt  \
--config_file {DIST}/preprocessing//uncased_L-12_H-768_A-12/bert_config.json  \
--max-sentences 32  --fast-stat-sync --max-update 900000 --update-freq 4  \
--valid-subset test --num-workers 4 \
--warmup-updates 10000  --total-num-update 1000000 --lr 0.0001  \
--weight-decay 0.01 --save-dir node1gpu4  \
--distributed-init-method tcp://{AD} \
--distributed-world-size 16 \
--distributed-gpus 4 \
--distributed-rank 8 \
2>&1 | node4gpu4_sub2.log
```

* 3-3) fourth node:
```
python3 ${DIST}/train.py  \
--task bert   --data {DIST}/preprocessing/test_128/ \
--dict {DIST}/preprocessing//uncased_L-12_H-768_A-12/vocab.txt  \
--config_file {DIST}/preprocessing//uncased_L-12_H-768_A-12/bert_config.json  \
--max-sentences 32  --fast-stat-sync --max-update 900000 --update-freq 4  \
--valid-subset test --num-workers 4 \
--warmup-updates 10000  --total-num-update 1000000 --lr 0.0001  \
--weight-decay 0.01 --save-dir node1gpu4  \
--distributed-init-method tcp://{AD} \
--distributed-world-size 16 \
--distributed-gpus 4 \
--distributed-rank 12 \
2>&1 | node4gpu4_sub3.log
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

## portable components (TODO)

## performance table (TODO)

### Notice and tips
> loading data may take a long time 

## License
this repository is MIT-licensed. It is created based on [fairseq](https://github.com/pytorch/fairseq), 
[NVIDIA-BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT), and 
[pytorch](https://github.com/pytorch/pytorch) 




