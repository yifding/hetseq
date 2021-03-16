import sys
from setuptools import setup, find_packages, Extension

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

if sys.platform == 'darwin':
    extra_compile_args = ['-stdlib=libc++', '-O3']
else:
    extra_compile_args = ['-std=c++11', '-O3']


class NumpyExtension(Extension):
    """Source: https://stackoverflow.com/a/54128391"""

    def __init__(self, *args, **kwargs):
        self.__include_dirs = []
        super().__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        import numpy
        return self.__include_dirs + [numpy.get_include()]

    @include_dirs.setter
    def include_dirs(self, dirs):
        self.__include_dirs = dirs


extensions = [
    NumpyExtension(
        'hetseq.data.data_utils_fast',
        sources=['hetseq/data/data_utils_fast.pyx'],
        language='c++',
        extra_compile_args=extra_compile_args,
    ),
]


setup(
    name='hetseq',
    version='1.0.1',
    author='Yifan Ding',
    author_email='dyf0125@gmail.com',
    url="https://github.com/yifding/hetseq",
    project_urls={
        "Documentation": "https://hetseq.readthedocs.io",
        "Medium Post": "https://towardsdatascience.com/training-bert-at-a-university-eedcf940c754",
    },
    description='Distributed GPU Training on Heterogeneous Infrastructureg',
    long_description=long_description,
    long_description_content_type='text/markdown',
    setup_requires=[
        'cython',
	'numpy',
        'setuptools>=18.0',
    ],
    install_requires=[
        'cython',
	'numpy',
	'h5py==2.10.0',
	'torch==1.6.0',
	'tqdm==4.36.1',
	'chardet==3.0.4',
	'idna==2.8',
	'python-dateutil==2.8.0',
	'sphinx-rtd-theme==0.5.0',
	'sphinx==3.2.1',
	'boto3==1.9.244',
	'torchvision==0.7.0',
	'datasets==1.1.3',
	'transformers==4.1.1',
	'seqeval==1.2.2',
    ],
    packages=find_packages(exclude=[]),
    ext_modules=extensions,
    zip_safe=False,
)
