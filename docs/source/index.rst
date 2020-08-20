.. HetSeq documentation master file, created by
   sphinx-quickstart on Mon Aug 17 16:33:04 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/yifding/hetseq

HetSeq's documentation!
==================================

HetSeq is a distributed neural network platiform designed to run on Heterogeneous Infrastructure with common scientific shared file system. It can be run directly on command line with SSH or task queen submission system without privilege or any extra packages. It takes care of the data index randomization and assignment to different GPUs in the multi-node and multi-GPU setting. Users can easily extend HetSeq to many other models with minimum effort.

.. note::
	
	HetSeq requires installation of `PyTorch <https://github.com/pytorch/pytorch>`__ with GPU support and `NCCL <https://github.com/NVIDIA/nccl>`__.


.. toctree::
   :maxdepth: 3
   :caption: Getting Started:

   installing
   distribute

.. toctree::
   :maxdepth: 3
   :caption: Running HetSeq:

   parameter
   examples

.. toctree::
   :maxdepth: 3
   :caption: Extending HetSeq:

   extending

.. toctree::
   :maxdepth: 3
   :caption: Reference:
    
   task
   dataset
   model
   optimizer
   lr_scheduler
   meters
   progress_bar	

.. toctree::
   :maxdepth: 3
   :caption: Support:

   support





Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
