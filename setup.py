#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='Bother',
    version='0.1',
    packages=find_packages(),
    scripts=['bother.py'],
    install_requires=['numpy', 'rasterio', 'pillow', 'elevation']
    package_data={},
    author='bunburya',
    author_email='',
    description='A script to produce heightmaps (primarily for OpenTTD) using real-world elevation data.',
    keywords='srtm elevation heightmap openttd'
    url='https://github.com/bunburya/bother',
    classifiers=[
        'License :: OSI Approved :: MIT License'
    ]
)
