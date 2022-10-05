from distutils.core import setup
from distutils.extension import Extension

import numpy

ext_modules=[
    Extension('image_retrieval_cython_utils', ['src/image_retrieval_cython_utils.c'], include_dirs=[numpy.get_include()]),
]

setup(
    name='image-retrieval-cython-utils',
    version='0.0.10',
    ext_modules=ext_modules,
    install_requires=[
        "numpy",
    ],
)
