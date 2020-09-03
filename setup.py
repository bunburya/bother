#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(
    name='Bother',
    version='0.1',
    packages=find_packages(),
    scripts=['bother'],
    install_requires=['numpy', 'rasterio', 'pillow', 'requests', 'appdirs'],
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
