import os
import re
from setuptools import find_packages
from setuptools import setup
import io

here = os.path.abspath(os.path.dirname(__file__))
try:
    # obtain version string from __init__.py
    # Read ASCII file with builtin open() so __version__ is str in Python 2 and 3
    with open(os.path.join(here, 'pytorch_ext', '__init__.py'), 'r') as f:
        init_py = f.read()
    version = re.search('__version__ = "(.*)"', init_py).groups()[0]
except Exception:
    version = ''
try:
    # obtain long description from README and CHANGES
    with io.open(os.path.join(here, 'README.rst'), 'r', encoding='utf-8') as f:
        README = f.read()
    with io.open(os.path.join(here, 'CHANGES.rst'), 'r', encoding='utf-8') as f:
        CHANGES = f.read()
except IOError:
    README = CHANGES = ''

install_requires = []
tests_require    = []

setup(
    name="Pytorch_Ext",
    version=version,
    description="Extention to Pytorch, including new layers, utility functions, etc. ",
    long_description="\n\n".join([README, CHANGES]),
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    keywords="",
    author="David Leon (Dawei Leng)",
    author_email="daweileng@outlook.com",
    url="https://github.com/david-leon/Pytorch_Ext",
    license="MIT",
    packages=find_packages(),
    include_package_data=False,
    zip_safe=False,
    )
