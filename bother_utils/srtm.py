"""Functions to download SRTM data from CGIAR website and generate a
TIF file for the desired coordinates using that data.
"""

import os
import math
import shutil
import zipfile
from typing import Optional, Union, Iterable, Tuple, Dict

import requests
import appdirs
import rasterio
import rasterio.merge
from rasterio.io import MemoryFile


ZIP_BASE_URL = 'http://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/'
ZIP_FNAME = 'srtm_{x:02d}_{y:02d}.zip'
TIF_FNAME = 'srtm_{x:02d}_{y:02d}.tif'

TILE_X_BOUNDS = (1, 72)
TILE_Y_BOUNDS = (1, 24)
TILE_SHAPE = (6000, 6000)
SRTM_NODATA = -32768
CACHE_DIR = appdirs.user_cache_dir('bother', appauthor=False)


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

def get_tif_fpath(x: int, y: int, cache_dir: str) -> str:
    return os.path.join(cache_dir, 'extracted', TIF_FNAME.format(x=x, y=y))

def download_zip(url: str, save_path: str, chunk_size: int = 1024):
    r = requests.get(url)
    r.raise_for_status()
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
    return save_path

def get_xy_components(lon: float, lat: float) -> Tuple[int, int]:
    """For a given longitude and latitude, returns the relevant
    components of the name of the zip file containing the SRTM data (for
    example "35" and "02" in "srtm_35_02.zip").
    """
    mod_lon = (int(math.floor(lon)) + 180) // 5 + 1
    mod_lat = (64 - int(math.floor(lat))) // 5
    return mod_lon, mod_lat

def get_all_xy_components(left: float, bottom: float, right: float, top: float) -> Iterable[Tuple[int, int]]:
    x_min, y_max = get_xy_components(left, bottom)
    x_max, y_min = get_xy_components(right, top)
    
    components = []
    
    for x in wrap_range(x_min, x_max+1, *TILE_X_BOUNDS):
        for y in wrap_range(y_min, y_max+1, *TILE_Y_BOUNDS):
            components.append((x, y))
    
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
    fnames = {}
    for x, y in get_all_xy_components(left, bottom, right, top):
        fnames[(x, y)] = ZIP_FNAME.format(x=x, y=y)
    return fnames

def fetch_all_zips(fnames: Dict[Tuple[int, int], str], cache_dir: str) -> Dict[Tuple[int, int], str]:
    """Takes a dict mapping xy pairs to zip filenames, downloads the
    relevant files if they are not already in the cache, and returns
    a dict mapping each xy pair to the absolute paths of the relevant
    zip file.
    """
    
    fpaths = {}
    cached = get_cached_files(cache_dir, '.zip')
    for xy in fnames:
        fname = fnames[xy]
        fpath = os.path.join(cache_dir, fname)
        if fname not in cached:
            url = ZIP_BASE_URL+fname
            print(f'Downloading {fname} from {url}.')
            fpath = download_zip(url, fpath)
            if fpath is None:
                print(f'Got 404 when downloading {fname}; there is probably no data available.')
                continue
        else:
            print(f'{fname} was cached in {cache_dir}.')
        fpaths[xy] = fpath
    return fpaths

def unzip_all(zip_files: Iterable[str], cache_dir: str) -> str:
    extract_dir = get_extract_dir(cache_dir)
    if not os.path.exists(extract_dir):
        os.mkdir(extract_dir)
    for fpath in zip_files:
        if fpath is not None:
            print(f'Extracting {fpath} to {extract_dir}.')
            with zipfile.ZipFile(fpath, 'r') as zf:
                zf.extractall(extract_dir)
    return extract_dir

def create_tif_file(left: float, bottom: float, right: float, top: float, to_file: Optional[str] = None,
                    cache_dir: str = CACHE_DIR, nodata: int = 0) ->  Union[str, MemoryFile]:
    """Create a TIF file using SRTM data  for the box defined by left,
    bottom, right, top.  If to_file is provided, saves the resulting
    file to to_file and returns the path; otherwise, creates a
    rasterio.io.MemoryFile and returns that.
    """
    
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    
    xy = get_all_xy_components(left, bottom, right, top)
    zip_fnames = {}
    for x, y in xy:
        zip_fnames[(x, y)] = ZIP_FNAME.format(x=x, y=y)
    zip_fpaths = fetch_all_zips(zip_fnames, cache_dir)
    unzip_all(zip_fpaths.values(), cache_dir)
    srcs = [rasterio.open(get_tif_fpath(x, y, cache_dir), 'r') for x, y in xy]
    print(f'Creating TIF file from following files: {[s.name for s in srcs]}.')
    #print(f'Heights are: {[s.height for s in srcs]}.')
    #print(f'Widths are: {[s.width for s in srcs]}.')
    profile = srcs[0].profile
    data, transform = rasterio.merge.merge(srcs, (left, bottom, right, top), nodata=nodata)
    for src in srcs:
        src.close()
    clear_cache(cache_dir, True)
    bands, height, width = data.shape   # No idea if this is the correct order for height and width, but they are both
                                        # the same so it doesn't matter in this case
    profile.update({
        'height': height,
        'width': width,
        'transform': transform
    })
    print(f'Created TIF file with dimensions {width}x{height}.') 
    if to_file:
        print(f'Writing TIF file to {to_file}.')
        with rasterio.open(to_file, 'w', **profile) as dst:
            dst.write(data)
        return to_file
    else:
        memfile = MemoryFile()
        with memfile.open(**profile) as dst:
            dst.write(data)
        return memfile

def clear_cache(cache_dir: str = CACHE_DIR, extracted_only: bool = False):
    if extracted_only:
        to_remove = get_extract_dir(cache_dir)
    else:
        to_remove = cache_dir
    try:
        shutil.rmtree(to_remove)
    except FileNotFoundError:
        pass
