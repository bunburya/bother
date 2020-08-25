#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script to detect lakes in a .ASC or .TIF DEM file and set them to a specified elevation.

We use a flood fill algorithm to detect contiguous sets of pixels that have exactly identical elevation.
This is similar to the algorithm (apparently) used by the MicroDEM software:
see https://freegeographytools.com/2007/modifying-the-terrain-reflectance-display-in-microdem.
"""

import numpy as np
import rasterio

def get_lake(data: np.ndarray, row: int, col: int, checked: np.ndarray, min_size: int):
    #if not (len(checked) % 1000):
    #    print(f'Checked {len(checked)} so far.')
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

def get_all_lakes(data: np.ndarray, min_size: int):
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

def convert(infile: str, outfile: str, fill_lakes_as: int, min_lake_size: int = 9):
    in_raster = rasterio.open(infile)
    data = in_raster.read(1)
    lakes = get_all_lakes(data, min_lake_size)
    print(f'Found {len(lakes)} lakes.')
    for lake in lakes:
        for row, col in lake:
            data[row, col] = fill_lakes_as
    out_raster = rasterio.open(
        outfile,
        'w',
        driver=in_raster.driver,
        width=in_raster.width,
        height=in_raster.height,
        count=in_raster.count,
        dtype=data.dtype,
        crs=in_raster.crs,
        transform=in_raster.transform
    )
    out_raster.write(data, 1)
    out_raster.close()

"""
def bump_up_elev(data: np.ndarray, min_elev: int):
    height, width = data.shape
    for c in range(width):
        if data[:,c].max() == 0.0:
            continue
        for r in range(height):
            if data[r, c] < min_elev:
                data[r, c] = min_elev
"""

def main(args):
    if not len(args) == 5:
        print('Usage: set_lake_elev.py <infile> <outfile> <fill_lakes_as> <min_lake_size>')
        print('infile:  File to read (.asc or .tif).')
        print('outfile:  File to output to (output file will be the same format as infile).')
        print('fill_lakes_as:  Integer, elevation to set lakes to.')
        print('min_lake_size:  Minimum area (in pixels) of a contiguous region of identical elevation that will be '
              'considered a lake. You may wish to play around with this figure until you get the right amount of lakes '
              'for your purposes.')
    else:
        convert(args[1], args[2], int(args[3]), int(args[4]))

if __name__ == '__main__':
    import sys
    main(sys.argv)
