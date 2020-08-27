#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""WIP."""

import logging
import math
import time
import argparse
from typing import Optional, Set, Tuple, List

import numpy as np
from PIL import Image

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import DatasetReader, DatasetWriter, MemoryFile
import elevation


MERCATOR = 'EPSG:4326'
PSEUDO_MERCATOR = 'EPSG:3857'

def get_tif_file(left: float, bottom: float, right: float, top: float, to_file: str = None) -> str:
    """Download SRTM data for the given bounds using the elevation
    library and return the path to the (temporary) file.
    """
    bounds = (left, bottom, right, top)
    print(f'Downloading SRTM data from CGIAR SRTM website for bounds: {bounds}.')
    print('The following output is from the elevation library.')
    if to_file is None:
        to_file = f'/tmp/othg_{time.time()}.tif'
    # NOTE:  This actually downloads SRTM4 data, but the elevation library hasn't
    # been updated to reflect the change from SRTM3 to SRTM4
    elevation.clip((left, bottom, right, top), output=to_file, product='SRTM3')
    print(f'SRTM data downloaded to {to_file}.')
    return to_file


def resample(memfile: MemoryFile, scale_factor: float) -> MemoryFile:
    """Resample raster by a factor of scale_factor.
        scale_factor > 1:  Upsample
        scale factor < 1:  Downsample
    """

    print(f'Resampling raster with scaling factor of {scale_factor}.')

    with memfile.open() as src:
        
        print(f'Source raster has shape {src.shape}.')
        
        # resample data to target shape
        height = int(src.height * scale_factor)
        width = int(src.width * scale_factor)

        data = src.read(
            out_shape=(src.count, height, width),
            resampling=Resampling.bilinear
        )
        # scale image transform
        transform = src.transform * src.transform.scale(width, height)

        kwargs = src.profile.copy()
        kwargs.update({
            'height': height,
            'width': width
        })
        dst_memfile = MemoryFile()
        with dst_memfile.open(**kwargs) as dst:
            for i in range(1, src.count+1):
                dst.write(data)
        
            print(f'Resampled raster has shape {dst.shape}.') 
        
        return dst_memfile

def reproject_raster(memfile: MemoryFile, src_crs: str = MERCATOR, dst_crs: str = PSEUDO_MERCATOR) -> MemoryFile:
    """Reproject raster with CRS src_crs to new CRS dst_crs."""
    
    print(f'Reprojecting raster from {src_crs} to {dst_crs}.')
    with memfile.open() as src:
        print(f'Source raster has shape {src.shape}.')
        transform, width, height = calculate_default_transform(src_crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.profile.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        dst_memfile = MemoryFile()

        with dst_memfile.open(**kwargs) as dst:
            for i in range(1, src.count+1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear
                )
        
            print(f'Reprojected raster has shape {dst.shape}.')
        
        return dst_memfile

def get_lake(data: np.ndarray, row: int, col: int, checked: np.ndarray, min_size: int) -> Optional[Set[Tuple[int, int]]]:
    """Check if the pixel at data[row, col] belongs to a lake and, if so,
    return the lake as a set of points.
    
    We use a flood fill algorithm to detect contiguous sets of pixels
    that have exactly identical elevation.  This is similar to the
    algorithm (apparently) used by the MicroDEM software: see
    https://freegeographytools.com/2007/modifying-the-terrain-reflectance-display-in-microdem
    """
    elev = data[row, col]
    candidates = {(row,col)}
    lake = set()
    while candidates:
        #print(f'# candidates: {len(candidates)}.')
        (_row,_col) = candidates.pop()
        #print(f'Got candidate {_row},{_col}.')
        if checked[_row,_col]:
            #print('Candidate checked already.')
            continue
        try:
            candidate_elev = data[_row, _col]
            #print(f'Candidate elev is {candidate_elev}.')
        except IndexError:
            checked[_row, _col] = 1
            #print('Candidate OOB.')
            continue
        if candidate_elev == elev:
            #print('Got a match.')
            lake.add((_row, _col))
            if row > 0:
                candidates.add((_row-1,_col))
            candidates.add((_row+1,_col))
            if col > 0:
                candidates.add((_row,_col-1))
            candidates.add((_row,_col+1))
        checked[_row, _col] = 1
    if len(lake) >= min_size:
    #    print(f'Found lake of size {len(lake)}.')
        return lake

def get_all_lakes(data: np.ndarray, min_size: int) -> List[Set[Tuple[int, int]]]:
    """Find all lakes in the data.  A lake is defined as a contiguous
    region of at least min_size pixels of the exact same elevation."""

    height, width = data.shape
    checked = np.zeros((height+2, width+2))
    lakes = []
    for c in range(width):
        if data[:,c].max() == 0.0:
            checked[:,c] = 1
            continue
        for r in range(height):
            # Do basic checks before calling function, to avoid function call overhead
            if (not checked[r, c]) and (data[r, c] > 0.0):
                lake = get_lake(data, r, c, checked, min_size)
                if lake is not None:
                    lakes.append(lake)
            checked[r, c] = 1
    return lakes

def set_lakes_to_elev(memfile: MemoryFile, fill_lakes_as: int, min_lake_size: int) -> MemoryFile:
    """Find all lakes in the data for a raster and set the elevation of
    the relevant pixels to fill_lakes_as.
    """
    
    print(f'Finding lakes with minimum size of {min_lake_size} and setting elevation to {fill_lakes_as}.')
    with memfile.open() as src:
        data = src.read(1)
        lakes = get_all_lakes(data, min_lake_size)
        print(f'Found {len(lakes)} lakes.')
        for lake in lakes:
            for row, col in lake:
                data[row, col] = fill_lakes_as
        
        dst_memfile = MemoryFile()
        kwargs = src.profile.copy()
        with dst_memfile.open(**kwargs) as dst:
            dst.write(data, 1)

        return dst_memfile

def raise_low_pixels(memfile: MemoryFile, max_no_raise: float = 0.0, noisy: bool = False) -> MemoryFile:
    """Detect very low (but above sea level) elevations in a raster and
    increase the elevation of the relevant pixels such that, when the
    file is converted to a greyscale image, those pixels will be given
    a value of 1, rather than rounded down to 0.
    """
    
    with memfile.open() as src:
        data = src.read(1)
        # Floor all values at 0
        data *= (data > 0)
        max_elev = data.max()
        min_visible = math.ceil(max_elev / 255) # Minimum value that will be scaled to 1 in a greyscale image
        if noisy:
            # Add a small value to each affected pixel's elevation, which is proportionate to that pixel's original
            # elevation.  This ensures that fixed areas do not have a uniform elevation (unless they originally had
            # a uniform elevation), so that they are not picked up by Microdem as lakes.
            min_visible = (data / 10) + min_visible
        data = np.where((data > max_no_raise) & (data < min_visible), min_visible, data)
    
        dst_memfile = MemoryFile()
        kwargs = src.profile.copy()
        with dst_memfile.open(**kwargs) as dst:
            dst.write(data, 1)
        
        return dst_memfile

def to_png_file(memfile: MemoryFile, to_file: str, floor_negative: bool = False):
    """Save raster as a greyscale PNG file to to_file.  If floor_negative
    is True, any elevation values below zero are set to zero.
    """
    
    print(f'Saving raster as PNG file to {to_file}.')
    with memfile.open() as src:
        data = src.read(1)
        if floor_negative:
            data *= (data > 0)
        max_elev = data.max()
        min_elev = data.min()
        scale_factor = 255 / (max_elev - min_elev)
        data = ((data - min_elev) * scale_factor).astype(np.uint8)
        im = Image.fromarray(data, mode='L')
        print(f'Image size is {im.size}.')
        im.save(to_file)

def error(msg: str):
    print(f'ERROR: {msg}', file=sys.stderr)
    sys.exit(1)

def parse_namespace(ns: argparse.Namepsace):
    
    # Check for errors
    if (ns.bounds is None) and (ns.infile is None):
        error('Must pass --bounds or --infile.')
    elif (ns.bounds is not None) and (ns.infile is not None):
        error('--bounds and --infile are mutually exclusive.')
    
    # Process request
    if ns.bounds:
        tif_file = get_tif_file(*ns.bounds, ns.cache)
    else:
        tif_file = ns.infile
    with open(tif_file, 'rb') as f:
        memfile = MemoryFile(f)
        if ns.scale is not None:
            if ns.scale == 0:
                error('0 is invalid value for scaling.')
            memfile = resample(memfile, ns.scale)
        if ns.raise_elev is not None:
            # TODO
        

def main(args):
    bounds = (float(i) for i in sys.argv[1:5])
    tmp_file = get_tif_file(*bounds)
    with open(tmp_file, 'rb') as f:
        memfile = MemoryFile(f)
        memfile = resample(memfile, 0.5)
        memfile = reproject_raster(memfile)
        memfile = set_lakes_to_elev(memfile, 0, 80)
        memfile = raise_low_pixels(memfile)
        to_png_file(memfile, args[5])

if __name__ == '__main__':
    import sys
    parser = argparse.ArgumentParser(description='Generate heightmaps from SRTM data, primarily for use in OpenTTD.')
    
    # TODO:  CRS arguments
    
    parser.add_argument('-if', '--infile', help='Specify a TIF file to use to generate the heightmap, rather than '
                                                'fetching SRTM data from the internet.')
    parser.add_argument('-b', '--bounds', type=float, nargs=4,, metavar=('LEFT', 'BOTTOM', 'RIGHT', 'TOP'),
                        help='Specify the bounds for which to download the SRTM data.')
    parser.add_argument('-c', '--cache', help='Save the generated TIF file so that it can be re-used.')
    parser.add_argument('-s', '--scale', type=float, help='Factor by which to scale the data prior to converting to '
                                                          'a PNG. Values lower than one will downsample the data; '
                                                          'values greater than one will upsample the data. Bilinear '
                                                          'interpolation is used.')
    parser.add_argument('-r', '--raise', type=float, nargs='?', const=0.0, metavar='ELEVATION', dest='raise_elev',
                        help='Raise pixels with low elevation so that they have a non-zero value in the resulting '
                             'greyscale image. If a numerical argument is provided, only pixels with elevations '
                             'above that value will be raised (default is 0).')
    parser.add_argument('-l', '--lakes', type=int, nargs='?', const=80,, metavar='SIZE',
                        help='Detect lakes (as contiguous regions of a minimum size with the exact same elevation) '
                             'and set their elevation to zero (so they are rendered as water in OpenTTD). If provided, '
                             'the integer argument determines the minimum number of pixels such an area must contain '
                             'in order to be considered a lake (default is 80).')
    parser.add_argument('outfile', help='The file to which the greyscale PNG image will be written.', required=True)
    ns = parser.parse_args()
    
