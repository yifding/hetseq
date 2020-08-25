************
Installation
************

Create Conda Virtual Environment
--------------------------------
Recommend to create and activate conda virtual environment with Python 3.7.4

.. code:: console

        $ conda create --name hetseq
        $ conda activate hetseq
        $ conda install python=3.7.4

Git Clone and Install Packages
--------------------------------------------

.. code:: console

	$ git clone https://github.com/yifding/hetseq.git
	$ cd /path/to/hetseq
	$ pip install -r requirements.txt 
	$ pip install --editable . 

Download BERT Processed File
----------------------------

(Required to run ``bert`` model)
Download data files including training corpus, model configuration, and BPE dictionary. Test corpus from `here <https://drive.google.com/file/d/1ZPJVAiV7PsewChi7xKACrjuniJ2N9Sry/view?usp=sharing>`__, full data from `this link <https://drive.google.com/file/d/1Vq_UO-T9345uYs8a7zloukGfhDXSDd2A/view?usp=sharing>`__. Download ``test_DATA.zip`` for test or ``DATA.zip`` for full run, unzip it and place the ``preprocessing/`` directory inside the package directory.

Download MNIST Dataset (deprecated)
-----------------------------------

(Required to run ``mnist`` model)
Download MNIST dataset from ``torchvision``, see example `here <https://github.com/pytorch/examples/blob/master/mnist/main.py#L114>`__.

.. code:: python

	from torchvision import datasets
	dataset1 = datasets.MNIST('../data', train=True, download=True)




