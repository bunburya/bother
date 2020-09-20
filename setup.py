#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

scripts = ['bother']

if os.name == 'nt':
    scripts.append(os.path.join('scripts', 'bother.bat'))
    
setup(
    name='Bother',
    version='1.0',
    packages=find_packages(),
    scripts=scripts,
    install_requires=['numpy', 'rasterio', 'pillow', 'requests', 'appdirs', 'tqdm'],
    package_data={},
    author='bunburya',
    author_email='',
    description='A script to produce heightmaps (primarily for OpenTTD) using real-world elevation data.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='srtm elevation heightmap openttd',
    url='https://github.com/bunburya/bother',
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: MIT License'
    ]
)
