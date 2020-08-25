#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script to detect very low (but above sea level) elevations in a .ASC or .TIF DEM file and
increase the elevation of the relevant pixels such that, when the file is converted to a
greyscale image, those pixels will be given a value of at 1, rather than rounded down to 0.
"""

import math
import numpy as np
import rasterio

def fix_elev(src_file: str, dst_file: str, noisy: bool = True):
    with rasterio.open(src_file) as src:
        meta = src.meta.copy()
        data = src.read(1)
        # Floor all values at 0
        data *= (data > 0)
        max_elev = data.max()
        min_visible = math.ceil(max_elev / 255) # Minimum value that will be scaled to 1 in a greyscale image
        print(f'Bumping low elevations to {min_visible}.')
        if noisy:
            # Add a small value to each affected pixel's elevation, which is proportionate to that pixel's original
            # elevation.  This ensures that fixed areas do not have a uniform elevation (unless they originally had
            # a uniform elevation), so that they are not picked up by Microdem as lakes.
            min_visible = (data / 10) + min_visible
            meta['dtype'] = np.float64
        data = np.where((data > 0) & (data < min_visible), min_visible, data)
    
    with rasterio.open(dst_file, 'w', **meta) as dst:
        dst.write(data, 1)
        

def main(args):
    if len(args) == 3:
        fix_elev(args[1], args[2], noisy=False)
    else:
        print('Usage: fix_elev <infile> <outfile>')
        print('infile:  File to read (.asc or .tif).')
        print('outfile:  File to output to (output file will be the same format as infile).')

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
