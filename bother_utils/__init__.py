import logging
import math
import time
import tempfile
import zipfile
from typing import Optional, Set, Tuple, List

import numpy as np
from PIL import Image

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
import elevation


"""Functions to fetch, manipulate and convert SRTM elevation data."""

ZIP_BASE_URL = 'http://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/'
ZIP_FNAME = 'srtm_{x:02d}_{y:02d}.zip'
TIF_FNAME = 'srtm_{x:02d}_{y:02d}.tif'
MERCATOR = 'EPSG:4326'

TILE_X_BOUNDS = (1, 72)
TILE_Y_BOUNDS = (1, 24)
TILE_SHAPE = (6000, 6000)
SRTM_NODATA = -32768


def wrap_range(start: int, end: int, min_val: int = 1, max_val: int = 72) -> Iterable[int]:
    i = start
    if not ((min_val <= start <= max_val) and (min_val <= end <= max_val)):
        raise ValueError('start and end must each be between min_val and max_val.')
    while i != end:
        if i > max_val:
            i = min_val
        yield i
        i += 1

def is_cached(fname: str, cache_dir: str) -> bool:
    try:
        return fname in os.listdir(cache_dir)
    except FileNotFoundError:
        return False

def get_cached_files(cache_dir: str, ext: str = '') -> Iterable[str]:
    try:
        return list(filter(lambda f: f.endswith(ext), os.listdir(cache_dir)))
    except FileNotFoundError:
        return []

def get_extract_dir(cache_dir: str) -> str:
    return os.path.join(cache_dir, 'extracted')

def get_zip_fpath(x: int, y: int, cache_dir: str) -> str:
    return os.path.join(cache_dir, ZIP_FNAME.format(x=x, y=y))

def get_tif_fpath(x: int, y: int, cache_dir: str) -> str:
    return os.path.join(cache_dir, 'extracted', TIF_FNAME.format(x=x, y=y))

def download_zip(url: str, save_path: str, chunk_size: int = 1024):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def get_xy_components(lon: float, lat: float) -> Tuple[int, int]:
    """For a given longitude and latitude, returns the relevant
    components of the name of the zip file containing the SRTM data (for
    example "35" and "02" in "srtm_35_02.zip").
    """
    mod_lon = (int(math.floor(lon)) + 180) // 5 + 1
    mod_lat = (64 - int(math.floor(lat))) // 5
    return mod_lon, mod_lat

def get_all_xy_components(left: float, bottom: float, right: float, top: float) -> Iterable[Tuple[int, int]]:
    
    y_min, y_min = get_xy_components(left, bottom)
    y_max, y_max = get_xy_components(right, top)
    
    components = []
    
    for x in wrap_range(x_min, x_max, *TILE_X_BOUNDS):
        for y in wrap_range(y_min, y_max, *TILE_Y_BOUNDS):
            components.append(x, y)
    
    return components

#def get_raster_data(x: int, y: int, cache_dir) -> np.ndarray:
#    tif_fpath = get_tif_fpath(x, y, cache_dir)
#    if os.path.exists(tif_fpath):
#        with rasterio.open(tif_fpath) as src:
#            return src.read(1)
#    else:
#        memfile = rasterio.io.MemoryFile()
#        with memfile.open(driver='GTiff', count=1, width=6000, height=6000, dtype=np.uint16

def get_all_zip_fnames(left: float, bottom: float, right: float, top: float,
                     cache_dir: Optional[str] = None) -> Dict[Tuple[int, int], str]:
    """Get zip file URLs for all tiles relevant to the area within the
    given bounds.
    """
    
    for x, y in get_all_xy_components(left, bottom, right, top):
        urls[(x, y)] = ZIP_FNAME.format(x=x, y=y)
    return urls

def fetch_all_zips(fnames: Dict[Tuple[int, int], str], cache_dir: str) -> Dict[Tuple(x, y), str]:
    fpaths = {}
    cached = get_cached_files(cache_dir, '.zip')
    for xy in fnames:
        fname = fnames[xy]
        fpath = os.path.join(cache_dir, fname)
        if fname not in cached:
            fpath = download_zip(ZIP_BASE_URL+fname, fpath)
        fpaths[xy] = fpath
    return fpaths

def unzip_all(zip_files: Iterable[str], cache_dir: str) -> str:
    extract_dir = get_extract_dir(cache_dir)
    if not os.path.exists(extract_dir):
        os.mkdir(extract_dir)
    for fpath in zip_files:
        if fpath is not None:
            with zipfile.ZipFile(fpath, 'r') as zf:
                zf.extractall(extract_dir)
    return extract_dir

def get_combined_size(left: float, bottom: float, right: float, top: float, cache_dir: str) -> Tuple[int, int]:
    
    left_floor = math.floor(left / 5) * 5
    bottom_floor = math.floor(bottom / 5) * 5
    
    
    transform = rasterio.transform.from_bounds(
    sw = get_xy_components(left, bottom)
    ne = get_xy_components(right, top)
    num_tiles = (ne[0] - sw[0]) * (sw[1] - ne[1])
    num_pixels = num_tiles * 6000 * 6000
        

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

def reproject_raster(memfile: MemoryFile, dst_crs: str, src_crs: str = MERCATOR) -> MemoryFile:
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
