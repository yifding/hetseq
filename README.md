# HetSeq: Distributed GPU Training on Heterogeneous Infrastructure
This is our coding implementation for the paper:

>Yifan Ding, Nicholas Botzer, Tim Weninger. 
HetSeq: Distributed GPU Training on Heterogeneous Infrastructure, Proc. of Association for the Advancement of Artificial Intelligence (AAAI) Innovative Application of Artificial Intelligence, February 2021.

Author: Yifan Ding (yding4@nd.edu)

**arxiv paper available**: https://arxiv.org/abs/2009.14783

**Documentation available**: https://hetseq.readthedocs.io 

**Medium towards data science Post:** [Training BERT at a University](https://towardsdatascience.com/training-bert-at-a-university-eedcf940c754)

Documentation includes [Distributed Setting](https://hetseq.readthedocs.io/en/master/distribute.html), [Scripts to Run HetSeq](https://hetseq.readthedocs.io/en/master/examples.html), [Extending HetSeq](https://hetseq.readthedocs.io/en/master/extending.html), [Parameter Explanation](https://hetseq.readthedocs.io/en/master/parameter.html) and [Code Reference](https://hetseq.readthedocs.io/en/master/task.html).

## Overview
HetSeq is a distributed neural network platiform designed to run on Heterogeneous Infrastructure with common scientific shared file system. 
It can be run directly on command line with SSH or task queen submission system without privilege or any extra packages. It takes care of the data index randomization and assignment to different GPUs in the multi-node and multi-GPU setting. Users can easily extend HetSeq to many other models with minimum effort.

HetSeq requires installation of [PyTorch](https://github.com/pytorch/pytorch) with GPU support and [NCCL](https://developer.nvidia.com/nccl).

## Installation
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
$ pip install -r requirements.txt 
$ pip install --editable . 
```

3) **To Run BERT:** Download data files including training corpus, model configuration, and BPE dictionary. Test corpus from [here](https://drive.google.com/file/d/1ZPJVAiV7PsewChi7xKACrjuniJ2N9Sry/view?usp=sharing), full data from [this link](https://drive.google.com/file/d/1Vq_UO-T9345uYs8a7zloukGfhDXSDd2A/view?usp=sharing). Download test_DATA.zip for test or DATA.zip for full run, unzip it and place the ```preprocessing/``` directory inside the package directory. Available corpus under ```preprocessing/```, 

* phase one of BERT training corpus : ```preprocessing/hdf5_lower_case_1_seq_len_128.../wikicorpus_en/```
* phase two of BERT training corpus : ```preprocessing/hdf5_lower_case_1_seq_len_512.../wikicorpus_en/```
* sample test for phase one : ```preprocessing/test128/```
* sample test for phase two : ```preprocessing/test512/```
* see [NVIDIA-pytorch-BERT](https://arxiv.org/abs/1810.04805), [google_original_BERT](https://github.com/google-research/bert) and [BERT paper](https://arxiv.org/abs/1810.04805) for more information.
* current provided is generated from [NVIDIA-pytorch-BERT](https://arxiv.org/abs/1810.04805) with wikipedia data (book data is not available)

4) Running HetSeq script is available at https://hetseq.readthedocs.io/en/master/examples.html, 

### Distributed Configuration
HetSeq can be executed on single GPU on a single node, multiple GPUs on a single node, or multiple GPUs across multiple nodes. Main logic is defined at [train.py](https://github.com/yifding/hetseq/blob/master/train.py#L213). 

* **--distributed-init-method**: defines an initialization. e.g.: "tcp://10.32.82.207:11111" (tcp for multiple nodes) or "file:///hetseq/communicate.txt" (shared file for multiple nodes).
* **--distributed-world-size**: total number of GPUs used in the training.
* **--distributed-gpus**: the number of GPUs on the current node.
* **--distributed-rank**: represents the rank/index of the first GPU used on current node. 




## Performance table
Running BERT on nodes with 4 GPUs each.
| nodes | GPUs | epochs | batch size |  steps  | avg. time per step | training time | training loss | expansion | speedup |
|:-----:|:----:|:------:|:----------:|:-------:|:------------------:|:-------------:|:-------------:|:---------:|:-------:|
|   1   |   4  |    5   |     128    | 267,139 |        2.60s       |     7.19d     |     0.026     |     1     |    1    |
|   2   |   8  |    5   |     256    | 133,570 |        2.69s       |     4.19d     |     0.028     |    0.86   |   1.72  |
|   4   |  16  |    5   |     512    |  66,785 |        2.794       |     2.23d     |     0.031     |    0.81   |   3.22  |
|   8   |  32  |    5   |     1024   |  33,393 |        3.126       |     1.21d     |     0.055     |    0.74   |   5.94  |

## Notice and tips
> loading BERT data takes a while. 

## Known issues
- [ ] currently not supporting continue training
- [ ] mnist datasets download does not support multiple GPUs

## future patch
- [ ] bert processing pipeline not included
- [ ] interface of datasets/transformers not included
- [ ] hetseq not supporting download from pip
- [ ] evaluation separate/combined not included
- [ ] fp16 support

## License
this repository is MIT-licensed. It is created based on [fairseq](https://github.com/pytorch/fairseq), 
[NVIDIA-BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT), and 
[pytorch](https://github.com/pytorch/pytorch) 

Please send us e-mail or leave comments on github if have any questions.


Copyright (c) 2020 Yifan Ding and [Weninger Lab](https://www3.nd.edu/~tweninge/)




