#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A script to produce heightmaps (primarily for OpenTTD) using real-world
elevation data.
"""

import sys
import logging
import math
import time
import argparse
import os
import os.path
import tempfile
from typing import Optional, Set, Tuple, List

import numpy as np
from PIL import Image

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import DatasetReader, DatasetWriter, MemoryFile
import elevation


MERCATOR = 'EPSG:4326'
PSEUDO_MERCATOR = 'EPSG:3857'
SRTM_NODATA = -32768

# Situations we need to consider:
# - High elevation all-land maps:  offset so that lowest point is 1
# - High elevation maps with lakes: offset so that lowest non-lake point is 1, set lakes to 0
# - Below sea level all-land maps: offset so that lowest point is 1
# - Below sea level maps with lakes:  offset so that lowest non-lake point is 1, set lakes to 0
# - Maps with sea and below sea level land:  set below sea level land to 1


### Functions to fetch, manipulate and convert SRTM elevation data

def get_tif_file(left: float, bottom: float, right: float, top: float, to_file: str) -> str:
    """Download SRTM data for the given bounds using the elevation
    library and return the path to the (temporary) file.
    """
    bounds = (left, bottom, right, top)
    print(f'Downloading SRTM data from CGIAR SRTM website for bounds: {bounds}.')
    print('=== The following output is from the elevation library. ===')
    # NOTE:  This actually downloads SRTM4 data, but the elevation library hasn't
    # been updated to reflect the change from SRTM3 to SRTM4
    elevation.clip((left, bottom, right, top), output=to_file, product='SRTM3')
    print('=== End output from elevation library. ===')
    print(f'SRTM data downloaded to {to_file}.')
    return to_file

def handle_nodata(memfile: MemoryFile, set_to: int = 0, nodata: int = SRTM_NODATA) -> MemoryFile:
    
    with memfile.open() as src:
        
        data = src.read(1)
        data = np.where(data == nodata, set_to, data)
        
        dst_memfile = MemoryFile()
        kwargs = src.profile.copy()
        with dst_memfile.open(**kwargs) as dst:
            dst.write(data, 1)

        return dst_memfile
    
def remove_sea(memfile: MemoryFile, min_elev: int = 1) -> MemoryFile:
    """Offset elevation data so that lowest value is equal to min_elev.
    Useful for when real-world data includes land below sea level but no
    sea (eg, an in-land NL map).
    
    By default,min_elev is 1, which means you may need to use this in
    conjunction with raise_low_pixels to actually render as land.
    """
    
    with memfile.open() as src:
        
        
        data = src.read(1)
        offset = -(data.min() - min_elev)
        
        print(f'Increasing elevation by {offset}.')
        data += offset
        print(f'lowest elevation is {data.min()}')
        
        dst_memfile = MemoryFile()
        kwargs = src.profile.copy()
        with dst_memfile.open(**kwargs) as dst:
            dst.write(data, 1)

        return dst_memfile

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

def set_lakes_to_elev(memfile: MemoryFile, min_lake_size: int, fill_lakes_as: int = 0) -> MemoryFile:
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

def raise_undersea_land(memfile: MemoryFile, raise_to: int = 1):
    """Raise land with a negative elevation (ie, land that is below sea
    level) to raise_to. Probably shouldn't be called before
    set_lakes_to_elev, otherwise the newly raised land will be picked up
    as a lake by that function.
    """
    
    with memfile.open() as src:
        data = src.read(1)
        
        print(f'Raising pixels of elevation < 0 to {raise_to}.')
        data = np.where((data < 0), raise_to, data)
    
        dst_memfile = MemoryFile()
        kwargs = src.profile.copy()
        with dst_memfile.open(**kwargs) as dst:
            dst.write(data, 1)
        
        return dst_memfile

def raise_low_pixels(memfile: MemoryFile, max_no_raise: float = 0.0, max_brightness: int = 255,
                     noisy: bool = False) -> MemoryFile:
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
        min_visible = math.ceil(max_elev / max_brightness) # Minimum value that will be rounded to 1 in a greyscale image
        print(f'Raising pixels of {max_no_raise} < elevation < {min_visible} to {min_visible}.')
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

def to_png(memfile: MemoryFile, set_negative: Optional[float] = None, max_brightness: int = 255):
    """Save raster as a greyscale PNG file to to_file.  If set_negative
    is set, any elevation values below zero are set to that value (which
    should be in the range 0-255).
    """
    
    print(f'Converting raster to PNG image.')
    with memfile.open() as src:
        data = src.read(1)
        if set_negative:
            # TODO: Fix
            data *= (data > 0)
        max_elev = data.max()
        min_elev = data.min()
        print(f'max/min {max_elev}/{min_elev}')
        scale_factor = max_brightness / (max_elev - min_elev)
        if min_elev > 1:
            # If everywhere on the map is above sea level, we should scale
            # such that the lowest parts of the map appear slightly above sea level.
            floor = min_elev + 1
        else:
            floor = min_elev
        data = ((data - floor) * scale_factor).astype(np.uint8)
        im = Image.fromarray(data, mode='L')
        print(f'Image size is {im.size}.')
        return im

crop_modes = {'nw', 'n', 'ne', 'w', 'c', 'e', 'sw', 's', 'se'}
def crop_image(im: Image, width: int, height: int, mode: str) -> Image:
    """Crop an image to width x height. Where in the image to crop to is
    determined by mode.
    """
    
    mode = mode.lower()
    
    print(f'Cropping image to {width}x{height} using mode {mode}.')
    
    #      left  top  right  bottom
    box = [None, None, None, None]
    
    if mode.startswith('n'):
        box[1] = 0
    elif mode.startswith('s'):
        box[1] = im.height - height
    else:
        box[1] = (im.height - height) // 2
    box[3] = box[1] + height
    
    if mode.endswith('w'):
        box[0] = 0
    elif mode.endswith('e'):
        box[0] = im.width - width
    else:
        box[0] = (im.width - width) // 2
    box[2] = box[0] + width
    
    return im.crop(box)

def scale_image(im: Image, width: int, height: int) -> Image:
    """Scale an image to width x height."""
    print(f'Scaling image to {width}x{height}.')
    return im.resize((width, height))
        
def png_to_file(im: Image, to_file: str):
    """Save image to file."""
    
    print(f'Saving PNG image tp {to_file}.')
    im.save(to_file)

###  Functions in relation to the command line interface

def error(msg: str):
    print(f'ERROR: {msg}', file=sys.stderr)
    sys.exit(1)

def check_namespace(ns: argparse.Namespace):
    """Check for errors in the arguments passed."""
    
    if (ns.bounds is None) and (ns.infile is None):
        error('Must pass --bounds or --infile.')
    elif (ns.bounds is not None) and (ns.infile is not None):
        error('--bounds and --infile are mutually exclusive.')
    
    if (ns.scale_data is not None) and ns.scale_data == 0:
        error('0 is invalid value for scaling.')
    
    if ns.crop:
        res, mode = ns.crop
        if mode.lower() not in crop_modes:
            error(f'Mode must be one of {crop_modes}.')
        res = res.split('x')
        try:
            int(res[0])
            int(res[1])
        except (IndexError, ValueError):
            error('Size for cropped image must be in form "WIDTHxHEIGHT", where WIDTH and HEIGHT are integers.')
    if ns.scale_image:
        res = ns.scale_image.split('x')
        try:
            int(res[0])
            int(res[1])
        except (IndexError, ValueError):
            error('Size for scaled image must be in form "WIDTHxHEIGHT", where WIDTH and HEIGHT are integers.')
    

def parse_namespace(ns: argparse.Namespace):
    """Parse the arguments passed and execute request."""

    if ns.bounds:
        if ns.tif:
            to_file = os.path.abspath(ns.tif)
        else:
            to_file = os.path.join(tempfile.gettempdir(), f'othg_{time.time()}.tif')
        tif_file = get_tif_file(*ns.bounds, to_file)
    else:
        tif_file = ns.infile
    with open(tif_file, 'rb') as f:
        memfile = handle_nodata(MemoryFile(f))
        if ns.scale_data is not None:
            memfile = resample(memfile, ns.scale_data)
        if ns.no_sea:
            memfile = remove_sea(memfile)
        if ns.epsg:
            memfile = reproject_raster(memfile, dst_crs=f'EPSG:{ns.epsg}')
        if ns.lakes:
            memfile = set_lakes_to_elev(memfile, ns.lakes)
        if ns.raise_undersea is not None:
            memfile = raise_undersea_land(memfile, ns.raise_undersea)
        if ns.raise_low is not None:
            memfile = raise_low_pixels(memfile, ns.raise_low, ns.max_brightness)
        im = to_png(memfile, not ns.raise_low, ns.max_brightness)
        if ns.crop:
            res, mode = ns.crop
            res = res.split('x')
            width = int(res[0])
            height = int(res[1])
            im = crop_image(im, width, height, mode)
        if ns.scale_image:
            res = ns.scale_image.split('x')
            width = int(res[0])
            height = int(res[1])
            im = scale_image(im, width, height)
        
        png_to_file(im, ns.outfile)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate heightmaps from NASA\'s Shuttle Radar Topography Mission '
                                                 'elevation data, primarily for use in OpenTTD.')
    
    # TODO:
    # - CRS arguments
    # - scaling and cropping
    
    parser.add_argument('-if', '--infile', help='Specify a TIF file to use to generate the heightmap, rather than '
                                                'fetching SRTM data from the internet.')
    parser.add_argument('-b', '--bounds', type=float, nargs=4, metavar=('LEFT', 'BOTTOM', 'RIGHT', 'TOP'),
                        help='Specify the bounds (coordinates of bottom-left and top-right points) for which to '
                             'download the SRTM data.')
    parser.add_argument('-t', '--tif', help='Save the generated TIF file (before processing) so that it can be re-used.',
                        metavar='FILE')
    parser.add_argument('-sd', '--scale-data', type=float, dest='scale_data', metavar='FACTOR',
                        help='Factor by which to scale the data prior to converting to a PNG. Values lower than one '
                             'will downsample the data; values greater than one will upsample the data. Bilinear '
                             'interpolation is used.')
    parser.add_argument('--epsg', type=int, metavar='CODE',
                        help='The EPSG code of the projection to use for the image. Default is 4326 (WGS 84).') 
    parser.add_argument('-l', '--lakes', type=int, nargs='?', const=80, metavar='SIZE',
                        help='Detect lakes (as contiguous regions of a minimum size with the exact same elevation) '
                             'and set their elevation to zero (so they are rendered as water in OpenTTD). If provided, '
                             'the integer argument determines the minimum number of pixels such an area must contain '
                             'in order to be considered a lake (default is 80).')
    parser.add_argument('-ru', '--raise-undersea', type=int, nargs='?', const=1, metavar='ELEVATION',
                        dest='raise_undersea', help='Raise pixels with an elevation below zero (ie, land that is below '
                                                    'sea level) to the given level.')
    parser.add_argument('-rl', '--raise-low', type=float, nargs='?', const=0.0, metavar='ELEVATION', dest='raise_low',
                        help='Raise pixels with low elevation so that they have a non-zero value in the resulting '
                             'greyscale image. If a numerical argument is provided, only pixels with elevations '
                             'above that value will be raised (default is 0).')
    parser.add_argument('-ns', '--no-sea', dest='no_sea', action='store_true',
                        help='Should be passed when the whole map is above sea level, so the lowest points are '
                             'not normalised to sea level (but just above sea level.')
    parser.add_argument('-mb', '--max-brightness', type=int, default=255, dest='max_brightness',
                        help='Set the maximum brightness in the greyscale PNG (ie, the brightness of the highest point '
                             'in the data). Should be between 1 and 255; a lower value will lead to a flatter map in '
                             'OpenTTD.')
    parser.add_argument('-c', '--crop', nargs=2, metavar=('WIDTHxHEIGHT', 'MODE'),
                        help='Crop the resulting image to WIDTH x HEIGHT. MODE determines which region of the image to '
                             'crop to and must be one of nw, n, ne, e, c, w, sw, s, se. Note that you may prefer to '
                             'do the cropping and scaling in your favourite image editor.')
    parser.add_argument('-si', '--scale-image', dest='scale_image', metavar='WIDTHxHEIGHT',
                        help='Scale the resulting image to WIDTH x HEIGHT. Note that you may prefer to do the cropping '
                             'and scaling in your favourite image editor.')
    parser.add_argument('outfile', help='The file to which the greyscale PNG image will be written.')
    ns = parser.parse_args()
    check_namespace(ns)
    parse_namespace(ns)
    
