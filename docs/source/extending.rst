********
Overview
********

To extend HetSeq to another model, one needs to define a new :doc:`Task </task>`  with corresponding :doc:`Model </model>`, :doc:`Dataset </dataset>`, :doc:`Optimizer </optimizer>` and :doc:`Learning Rate Scheduler </lr_scheduler>`. A ``MNIST`` example is given with all the extended classes. Pre-defined ``optimizers``, ``Learning Rate Scheduler``, ``datasets`` and ``models`` can be reused in other applications.

Task
----
For each individual application, task is the basic unit. Defined by ``class Task`` in ``Task.py``. ``datasets`` is stored and load in a dictionary manner. Define a child class of ``Task`` to define a new task, necessary function is to define ``Model`` (in ``def build_model``), ``Dataset`` (in ``def load_dataset``). 

.. code:: python

	class Task(object):
		def __init__(self, args):
       			self.args = args
        		self.datasets = {}
        		self.dataset_to_epoch_iter = {}
		
		def build_model(self, args):
        		raise NotImplementedError

		def load_dataset(self, split, **kwargs):
        		"""Load a given dataset split.
        		Args:
            		split (str): name of the split (e.g., train, valid, test)
        		"""
        		raise NotImplementedError

Dataset
-------

Dataset should be defined as a child class of ``torch.utils.data.Dataset`` to be compatible with ``torch.utils.data.dataloader``. Need to define 
	* ``__getitem__`` (get item), 
	* ``__len__`` (total length of the datset), 
	* ``ordered_indices`` (index used to split and assignment to different GPUs, 
	* ``np.arange(len(self))``), 
	* ``num_tokens`` (total tokens in a instance, ``1`` for image model), 
	* ``collater`` (collater function to combined the output of ``__getitem__``, typically use ``torch.utils.data.dataloader.default_collate``)
	* ``set_epoch`` (``pass``) function in the class. 

See following ``MNIST example`` for more information.


.. note::

	In our implementation, each process/GPU has its own dataset and dataloader. When dataset is small (like MNIST example), the dataset can be put into ``__init__`` function. However, if the dataset is large (like BERT example or ImageNet), the dataset can not be loaded into memory at once, then the loading process should be defined inside ``__getitem__`` function. 


Model
-----

Model should be defined as a child class of ``torch.nn.Module``. By default, the model should output a loss function. This is compatible with the ``def train_step(self, sample, model, optimizer, ignore_grad=False)`` function inside ``class Task``. One can change the logic but need to fit the ``train_step``. 


Optimizer
---------

Optimizer in distributed data parallel (DDP) has to consider manipulate gradients and learning rates. In our implementation, optimizer (``class _Optimizer(object):``) is defined as a higher level class than ``torch.optim.Optimizer`` to include other parameters to be recorded. For example, the ``Adam`` optimizer provided in HetSeq, has initial learning rate:``lr``, beta1 and beta2: ``betas``, epsilon ``eps`` to avoid normalize by ``0`` and weight decay ``weight_decay``.

.. code:: python

	class _Adam(_Optimizer):
    		def __init__(self, args, params):
        		super().__init__(args)

        		self._optimizer = Adam(params, **self.optimizer_config)

    		@property
    		def optimizer_config(self):
        		"""
        		Return a kwarg dictionary that will be used to override optimizer
        		args stored in checkpoints. This allows us to load a checkpoint and
        		resume training using a different set of optimizer args, e.g., with a
        		different learning rate.
        		"""
        		return {
            			'lr': self.args.lr[0],
            			'betas': eval(self.args.adam_betas),
            			'eps': self.args.adam_eps,
            			'weight_decay': self.args.weight_decay,
        		}




Learning Rate Scheduler
-----------------------

In HetSeq, common ``PolynomialDecayScheduler`` is provided and compatible to ``BERT model`` and ``MNIST model``. 
Other learning rate scheduler can be easily extended by providing ``step_update`` and ``step`` function. 


*************
MNIST example
*************

MNIST example is adapted from `PyTorch mnist example <https://github.com/pytorch/examples/tree/master/mnist>`__.  It is convolutional neural network model for image classification. We adapt the original datasets, model and data loader to be compatible to HetSeq. 


Task
----

.. code:: python

	class MNISTTask(Task):
    		def __init__(self, args):
        		super(MNISTTask, self).__init__(args)

    	@classmethod
    	def setup_task(cls, args, **kwargs):
        	"""Setup the task (e.g., load dictionaries).
        	Args:
            		args (argparse.Namespace): parsed command-line arguments
        	"""
	        return cls(args)

    	def build_model(self, args):
        	model = MNISTNet()
        	return model

    	def load_dataset(self, split, **kwargs):
        	"""Load a given dataset split.
        	Args:
         		split (str): name of the split (e.g., train, valid, test)
        	"""
        	path = self.args.data

        	if not os.path.exists(path):
            		raise FileNotFoundError(
                		"Dataset not found: ({})".format(path)
            		)

        	if os.path.isdir(path):
            		if os.path.exists(os.path.join(path, 'MNIST/processed/')):
                		path = os.path.join(path, 'MNIST/processed/')
            	elif os.path.basename(os.path.normpath(path)) != 'processed':
                	datasets.MNIST(path, train=True, download=True)
                	path = os.path.join(path, 'MNIST/processed/')

        	files = [os.path.join(path, f) for f in os.listdir(path)] if os.path.isdir(path) else [path]
        	files = sorted([f for f in files if split in f])

        	assert len(files) == 1, "no suitable file in split ***{}***".format(split)

        	dataset = MNISTDataset(files[0])

        	print('| loaded {} sentences from: {}'.format(len(dataset), path), flush=True)

        	self.datasets[split] = dataset
        	print('| loading finished')


Dataset
-------
.. code:: python

	class MNISTDataset(torch.utils.data.Dataset):
    		def __init__(self, path):
        		self.data = None
        		self.path = path
        		self.read_data(self.path)
        		self.transform = transforms.Compose([
            			transforms.ToTensor(),
            			transforms.Normalize((0.1307,), (0.3081,))
        		])


    		def read_data(self, path):
        		self.data = torch.load(path)
        		self._len = len(self.data[0])
        		self.image = self.data[0]
        		self.label = self.data[1]


    		def __getitem__(self, index):
        		img, target = self.image[index], int(self.label[index])
        		img = Image.fromarray(img.numpy(), mode='L')
        		img = self.transform(img)
        		return img, target

    		def __len__(self):
       			return self._len

    		def ordered_indices(self):
        		"""Return an ordered list of indices. Batches will be constructed based
        		on this order."""
        		return np.arange(len(self))

   		def num_tokens(self, index: int):
        		return 1

    		def collater(self, samples):
		        if len(samples) == 0:
            			return None
        		else:
            			return default_collate(samples)

    		def set_epoch(self, epoch):
        		pass

Model
-----

.. code:: python

	class MNISTNet(nn.Module):
   		def __init__(self):
        		super(MNISTNet, self).__init__()
        		self.conv1 = nn.Conv2d(1, 32, 3, 1)
        		self.conv2 = nn.Conv2d(32, 64, 3, 1)
        		self.dropout1 = nn.Dropout2d(0.25)
        		self.dropout2 = nn.Dropout2d(0.5)
        		self.fc1 = nn.Linear(9216, 128)
        		self.fc2 = nn.Linear(128, 10)

    		def forward(self, x, target, eval=False):
        		x = self.conv1(x)
        		x = F.relu(x)
        		x = self.conv2(x)
        		x = F.relu(x)
        		x = F.max_pool2d(x, 2)
        		x = self.dropout1(x)
        		x = torch.flatten(x, 1)
        		x = self.fc1(x)
        		x = F.relu(x)
        		x = self.dropout2(x)
        		x = self.fc2(x)
        		output = F.log_softmax(x, dim=1)
        		loss = F.nll_loss(output, target)
        		return loss

Running Script
--------------

See :doc:`running script </examples>` for details.


