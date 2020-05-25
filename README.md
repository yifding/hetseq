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

## Recommended install step with conda
* conda create --name distribute_bert
* conda activate distribute_bert
* conda install python=3.7.4

* git clone url (TODO)
* cd url_directory (TODO)
* pip install -r requirement.txt 
* pip install --editable . 
* download extra data from google drive [here](https://drive.google.com/file/d/) (TODO)
* unzip file in the url_directory (TODO)

## data stats and source (TODO)

## Example to Run the Codes
### single GPU
### multiple GPU on a single Node
### multiple GPU on multiple Nodes

### Notice
> loading data may take a long time 

## License
this repository is MIT-licensed. It is created based on [fairseq](https://github.com/pytorch/fairseq), 
[NVIDIA-BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT), and 
[pytorch](https://github.com/pytorch/pytorch) 




