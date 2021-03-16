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
    version='1.0.0',
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
        'torch',
        'tqdm',
    ],
    packages=find_packages(exclude=[]),
    ext_modules=extensions,
    zip_safe=False,
)
