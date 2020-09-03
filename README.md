# Bother - Bunburya's OpenTTD Heightmap Generator

## Introduction

Bother is a Python script to generate heightmaps from real-world elevation data. It was written with the intention of using the heightmaps to create playable maps in [OpenTTD](https://www.openttd.org/) (but there is no reason the heightmaps it generates can't be used for other purposes).

The elevation data used for the heightmaps is taken from NASA's Shuttle Radar Topography Mission data, which covers between 60° north and 56° south latitude (over 80% of the Earth's land surface).

## Installation

At the outset, you should note that Bother has been developed and tested on Linux.  There is no reason why it shouldn't work on Windows and Mac, but it hasn't been tested out on those platforms.  If you encounter any problems trying to install or use Bother on any platform, please open an issue [on GitHib](https://github.com/bunburya/bother).

Bother uses Python 3 so make sure you have it installed on your computer. 

Then, you can install Bother from PyPI with `pip3 install bother`.  If you don't have pip installed, you can download the source from GitHub and run `python3 setup.py install` in the source folder.   

## Basic usage

Bother is a command line utility, designed to be called from the command line.

The most basic way to use Bother is by providing the bounding coordinates (latitude/longitude) of the area you want to map, using the `--bounds` option, followed by the name of the file to which you want to save the resulting PNG greyscale image.  When you use Bother in this way, you can also optionally provide a `--output-tif` option, followed by the path to which you want to save the file (a GeoTIFF file, with extension `.tif`) containing the SRTM data.

Instead of using the `--bounds` option (which will fetch elevation data from the internet), you can alternatively provide Bother with a GeoTIFF file containing the relevant data, using the `--infile-tif` option followed by the path to the relevant file.  This file can, for example, be the GeoTIFF file you got when you provided the `--output-tif` option previously.

The above options (and most of the options we mention below) also have short-form equivalents.  Initialise Bother with the `--help` or `-h` option to get a full description of the various options available.

## Additional options

By default, Bother simply takes elevation data and converts it to integers the range 0-255 (or, if the entire map is above sea level, 1-255).  There are a few additional options which you can provide to Bother to tell it to perform some additional manipulation of the elevation data.  Depending on the circumstances, these options may result in better heightmaps for your purposes.

### Resampling data

Providing the `--scale-data` option followed by a number will resample the elevation, data using bilinear interpolation to fill in any gaps.  A number less than 1 will downsample the data, resulting in a smaller file (and speeding up processing time); a number greater than one will upsample the data, resulting in a larger file.

### Reprojecting

By default, the data will be displayed using the [WGS 84](https://en.wikipedia.org/wiki/World_Geodetic_System#WGS84) (EPSG:4326) spatial reference system for mapping latitude / longitude coordinates to pixels.  If you would like to use some other spatial reference system, you can provide the `--epsg` option followed by the numerical EPSG code of the relevant system.  For example, providing `--epsg 3857` would result in the data being projected using the [Web Mercator projection](https://en.wikipedia.org/wiki/Web_Mercator_projection) that is used by most popular web-based mapping applications.

### Raising low (but above sea level) pixels

By default, because the elevation of every point is converted to an integer between 0 and 255 when converting the SRTM data to a greyscale image, some very low elevations will be rounded to 0.  Some loss of precision is inevitable when converting elevation data to a greyscale image, and generally this probably won't be a problem.  However, because this may result in low land being rendered by OpenTTD as sea, you may occasionally find noticable defects in the resulting map, such as peninsulas being rendered as islands, or islands not being rendered at all.  If you pass the `--raise-low` option, any elevation values that are greater than 0 but small enough that they would normally be rounded down to 0 will instead be rounded up to 1, thus ensuring that they are rendered as land in OpenTTD.

If you provide a numerical argument directly after the `--raise-low` option, then only elevation values above that value will be rounded up to 1.  For example, if you provide `--raise-low 1`, an elevation of 1 will be rounded down to 0 (assuming it would have been rounded down in any event), whereas an elevation above 1 but that would otherwise have been rounded down to 0 will be rounded up. 

### Raising below sea level pixels

OpenTTD cannot comprehend the concept of dry land below sea level; the lowest elevation it understands is sea level, and anything at sea level is rendered as sea.  Accordingly, if the area you are mapping contains both sea and dry land that is below sea level (ie, elevation < 0), you can provide the `--raise-undersea` option to set the elevation of the below sea level points to some value above 0, so that it will be rendered as land in OpenTTD.  By default, the `--raise-undersea` option will give those points an elevation of 1 metre above sea level, in which case you may need to also call `--raise-low` to ensure that they are actually rendered as land.  Alternatively, you can provide some numerical argument to the `--raise-undersea` option to set the elevation of the relevant points to a custom value. 

### Lake detection

Providing the `--lakes` option will try to detect lakes in the data and set them to 0 elevation, so that OpenTTD renders them as water.  It does this by detecting contiguous regions of pixels that have the exact same elevation.  Only regions containing a minimum number of pixels will be flagged as lakes; by default, this number is 80, but you can change that by following the `--lakes` option with an integer number.

### Maximum brightness (elevation)

As mentioned above, by default Bother will scale elevation data so that the lowest elevation is given a value of 0 or 1, and the highest elevation is given a value of 255.  OpenTTD will render a pixel with a "brightness" of 255 as the highest elevation possible, given the game settings (specifically, the "maximum map height" setting you specify when starting a game).  The lowest maximum map height is 15, meaning that the highest elevation in the SRTM data will be rendered as having a height of 15 in-game.  If you are trying to map very flat land, you may wish to provide the `--max-brightness` option with an integer value of somewhere between 1 and 255.  The highest point in your SRTM data will then be given a value of whatever number you have provided, rather than 255, and the rest of the values will be scaled accordingly.  

## Cropping and scaling

By default, the dimensions of the PNG image output by Bother will depend on the region specified and the projection used.  To use the PNG file in OpenTTD, it is a good idea to resize it to dimensions that are supported by OpenTTD.  The best way to do this is probably to open up the PNG file in your favourite image editor, where you can crop and scale with the benefit of being able to see the image you are working with.  However, Bother also provides some basic options for cropping and scaling.

Providing the `--input-png` option followed by a path to a PNG file will tell Bother to load the PNG file in order to perform cropping and scaling, rather than creating a new PNG file from SRTM data.

### Cropping

You can tell Bother to crop your image by providing the `--crop` option, followed by the resolution (in the form `WIDTHxHEIGHT`, where `WIDTH` and `HEIGHT` are integers) and the mode, which will tell Bother which part of the image to keep.  The following modes are available:


        nw  n   ne
    
        w   c   e

        sw  s   se


So, for example, a mode of `nw` will tell Bother to keep the top left (north-west) part of the image, and a mode of `c` will tell Bother to keep the middle of the image, cropping all the borders equally.

If you specify a dimension that is larger than the corresponding dimension of the input image, no cropping will be performed along that dimension, eg, trying to crop a 500x750 image to 600x700 will instead crop the image to 500x700. 

### Scaling

Provide the `--scale-image` option along with the desired resolution (in `WIDTHxHEIGHT` form, as above) to scale the output image.

## Examples

Here, we call Bother to create a PNG file of the rectangle bounded by 51.296094, -11.017116 to 55.596810, -4.403352 (roughly, the island of Ireland, with a bit of Scotland, Wales and the Isle of Man).  We tell Bother to downsample the data by about half, reproject the data in the Web Mercator projection, detect lakes, and raise any low-lying pixels.

`bother --scale-data 0.5 --epsg 3857 --lakes --raise-low --bounds 51.296094 -11.017116 55.596810 -4.403352 ireland.png`

If all goes well, you should see (among other output) a message telling you that the output image size is 3151x3531.  We want the image to be 4096x4096 so we can play it as a large map in OpenTTD.  So we call Bother again, this time telling it to take the previously created PNG file, crop it to 3151x3151 (keeping the centre of the image, ie, removing the edges) and scale it to 4096x4096.

`bother --infile-png ireland.png --crop 3151x3151 c --scale-image 4096x4096 ireland.png`

The resulting PNG file and a screenshot of the resulting OpenTTD map are available in the "examples" folder on GitHub.
