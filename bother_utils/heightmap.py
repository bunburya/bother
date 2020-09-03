import logging
import math
import time
import tempfile
import zipfile
from typing import Optional, Set, Tuple, List

import numpy as np
from PIL import Image

"""Functions to manipulate TIF files and convert them into heightmaps."""


import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile


DEFAULT_CRS = 'EPSG:4326' # Mercator


#def handle_nodata(memfile: MemoryFile, set_to: int = 0, nodata: int = SRTM_NODATA) -> MemoryFile:
#    
#    with memfile.open() as src:
#        
#        data = src.read(1)
#        data = np.where(data == nodata, set_to, data)
#        
#        dst_memfile = MemoryFile()
#        kwargs = src.profile.copy()
#        with dst_memfile.open(**kwargs) as dst:
#            dst.write(data, 1)
#
#        return dst_memfile
    
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

def reproject_raster(memfile: MemoryFile, dst_crs: str, src_crs: str = DEFAULT_CRS) -> MemoryFile:
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
        (_row,_col) = candidates.pop()
        if checked[_row,_col]:
            continue
        try:
            candidate_elev = data[_row, _col]
        except IndexError:
            checked[_row, _col] = 1
            continue
        if candidate_elev == elev:
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

def to_png(memfile: MemoryFile, zero_floor: bool = False, max_brightness: int = 255):
    """Save raster as a greyscale PNG file to to_file.  If set_negative
    is set, any elevation values below zero are set to that value (which
    should be in the range 0-255).
    """
    
    print(f'Converting raster to PNG image.')
    with memfile.open() as src:
        data = src.read(1)
        if zero_floor:
            data *= (data > 0)
        max_elev = data.max()
        min_elev = data.min()
        scale_factor = max_brightness / (max_elev - min_elev)
        if min_elev > 0:
            # If everywhere on the map is above sea level, we should scale
            # such that the lowest parts of the map appear slightly above sea level.
            floor = min_elev + 1
        else:
            floor = min_elev
        data = ((data - floor) * scale_factor).astype(np.uint8)
        im = Image.fromarray(data, mode='L')
        width, height = im.size
        print(f'Image size is {width}x{height}.')
        return im

crop_modes = {'nw', 'n', 'ne', 'w', 'c', 'e', 'sw', 's', 'se'}
def crop_image(im: Image, width: int, height: int, mode: str) -> Image:
    """Crop an image to width x height. Where in the image to crop to is
    determined by mode.
    """
    
    mode = mode.lower()
    
    width = min(width, im.width)
    height = min(height, im.height)
    
    print(f'Cropping image to {width}x{height} using mode {mode}.')
    
    #       left top right bottom
    box =  [0, 0, 0, 0]
    
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
    
    print(f'Saving PNG image to {to_file}.')
    im.save(to_file)
