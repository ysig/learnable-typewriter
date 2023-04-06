#!/usr/bin/env python
from setuptools import find_packages
from distutils.core import setup

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

setup(name='learnable_typewriter',
      version='0.1.0',
      description='The Learnable Typewriter',
      author='Ioannis Siglidis [Imagine / LIGM / ENPC]',
      author_email='ioannis.siglidis@enpc.fr',
      url='https://www.python.org/sigs/distutils-sig/',
      license="MIT",
      python_requires='>=3.5,',
      install_requires=INSTALL_REQUIRES,
      packages=find_packages(),
      scripts=['scripts/train.py', 'scripts/eval.py'],
     )
