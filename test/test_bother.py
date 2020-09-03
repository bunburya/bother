#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append(sys.argv[1])

import unittest
import os
import shutil

import numpy as np
from PIL import Image, ImageChops, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from bother_utils.heightmap import *
from bother_utils.srtm import *


EXAMPLES_DIR = 'examples'
TEST_DIR = 'test_data'

test_coords = {
    'mallorca': (39.195468, 2.272535, 39.998336, 3.584351),
    'ireland': (51.296094, -11.017116, 55.596810, -4.403352),
    'constance': (47.321556, 8.830936, 47.980224, 9.860904),
    'titicaca': (-16.970092, -70.246331, -15.032316, -68.334710),
    'alps': (46.408240, 9.555657, 46.975534, 10.378602)
}


# No options provided
MALLORCA_TIF = os.path.join(EXAMPLES_DIR, 'mallorca.tif')
MALLORCA_PNG = os.path.join(EXAMPLES_DIR, 'mallorca.png')



class BotherTestCase(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print('SETTING UP CLASS')
        if not os.path.exists(TEST_DIR):
            os.mkdir(TEST_DIR)
        print(os.listdir())
    
    @classmethod
    def tearDownClass(cls):
        print('TEARING DOWN CLASS')
        shutil.rmtree(TEST_DIR)
    
    def _assert_images_equal(self, im1, im2):
        np.testing.assert_array_equal(
            np.array(im1.getdata()),
            np.array(im2.getdata())
        )
    
    def _assert_image_files_equal(self, fpath1, fpath2):
        with Image.open(fpath1) as im1:
            with Image.open(fpath2) as im2:
                self._assert_images_equal(im1, im2)
    
    def test_1_srtm_tif(self):
        """Test the basic downloading of TIF files."""
        
        for eg in test_coords:
            bottom, left, top, right = test_coords[eg]
            tif_file = os.path.join(TEST_DIR, f'{eg}.tif')
            eg_tif_file = os.path.join(EXAMPLES_DIR, f'{eg}.tif')
            create_tif_file(left, bottom, right, top, os.path.abspath(tif_file))
            self._assert_image_files_equal(tif_file, eg_tif_file)

    def test_2_resize(self):
        """Test basic cropping and scaling."""
        
        with open(os.path.join(TEST_DIR, 'mallorca.tif'), 'rb') as f:
            memfile = MemoryFile(f)
            with memfile.open() as src:
                data = src.read(1)
                shape = data.shape
            with memfile.open() as src:
                data = src.read(1)
            im = to_png(memfile, True, 255)
        im = crop_image(im, 1574, 787, 'c')
        im1 = scale_image(im, 2048, 1024)
        with Image.open(os.path.join(EXAMPLES_DIR, 'mallorca_no_processing_resized.png')) as im2:
            self._assert_images_equal(im1, im2)
    
    def test_3_downsample_lakes_raise_low_flat(self):
        """Test data resampling, lake detection, raising low pixels and flattening."""
        
        with open(os.path.join(TEST_DIR, 'ireland.tif'), 'rb') as f:
            memfile = MemoryFile(f)
            memfile = resample(memfile, 0.5)
            memfile = set_lakes_to_elev(memfile, 100)
            memfile = raise_low_pixels(memfile)
            im1 = to_png(memfile)
        with Image.open(os.path.join(EXAMPLES_DIR, 'ireland_resampled_lakes_low_raised.png')) as im2:
            self._assert_images_equal(im1, im2)
        
        with open(os.path.join(TEST_DIR, 'ireland.tif'), 'rb') as f:
            memfile = MemoryFile(f)
            memfile = resample(memfile, 0.5)
            memfile = set_lakes_to_elev(memfile, 100)
            memfile = raise_low_pixels(memfile, max_brightness=170)
            im1 = to_png(memfile, False, 170)
        with Image.open(os.path.join(EXAMPLES_DIR, 'ireland_resampled_lakes_low_raised_flat.png')) as im2:
            self._assert_images_equal(im1, im2)
            
    def test_4_lakes_all_above_sea_epsg_upsample(self):
        """Test a map where everything is above sea level (including one map where there are lakes).
        Also tests reprojection and upsampling of data.
        """
                
        with open(os.path.join(TEST_DIR, 'titicaca.tif'), 'rb') as f:
            memfile = MemoryFile(f)
            memfile = reproject_raster(memfile, dst_crs='EPSG:3857')
            memfile = set_lakes_to_elev(memfile, min_lake_size=80)
            im1 = to_png(memfile)
        with Image.open(os.path.join(EXAMPLES_DIR, 'titicaca_lakes_3857.png')) as im2:
            self._assert_images_equal(im1, im2)
        
        with open(os.path.join(TEST_DIR, 'alps.tif'), 'rb') as f:
            memfile = MemoryFile(f)
            im1 = to_png(memfile)
        with Image.open(os.path.join(EXAMPLES_DIR, 'alps.png')) as im2:
            self._assert_images_equal(im1, im2)

if __name__ == '__main__':
    unittest.main(argv=['hi'])
