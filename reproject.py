#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""WIP."""

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import DatasetReader, DatasetWriter

from PIL import Image

MERCATOR = 'EPSG:4326'
PSEUDO_MERCATOR = 'EPSG:3857'

def _reproject(src_file: str, dst_file: str, src_crs: str = MERCATOR, dst_crs: str = PSEUDO_MERCATOR):
    
    with rasterio.open(src_file) as src:
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(dst_file, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear
                )

def to_png(src_file: str, to_file: str):
    with open(src_file) as src:
        im = Image.new('L', (src.width, src.height))
        elev_data = src.read(1)
        # Floor all values at 0
        data *= (data > 0)
        max_elev = elev_data.max()
        min_elev = min_elev.min()
        for row in range(src.height):
            for col in range(src.width):
                elev = elev_data[row, col]
                # TODO: Finish
    
if __name__ == '__main__':
    import sys
    _reproject(sys.argv[1], sys.argv[2])
