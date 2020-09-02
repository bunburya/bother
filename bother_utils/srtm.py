#!/usr/bin/env python
# -*- coding: utf-8 -*-

BASE_URL = 'http://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF'

def get_zip_url(lat: float, lon: float) -> str:
    
    return f'{BASE_URL}/srtm_{lon:02d}_{lat:02d}.zip'

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

def get_transform(left: float, bottom: float, right: float, top: float) -> affine.Affine:
    
    sw = get_xy_components(left, bottom)
    ne = get_xy_components(right, top)

    height = (sw[1] - ne[1] + 1) * 6000
    width = (ne[0] - sw[0] + 1) * 6000
    
    return rasterio.transform.from_bounds(
        math.floor(left / 5) * 5,
        math.floor(bottom / 5) * 5,
        math.ceil(right / 5) * 5,
        math.ceil(top / 5) * 5,
        width, height
    )

# Process for creating combined file:
# - Get transform from bounds.
# - Merge data from all files into one giant array.
# - Use transform to figure out row and cols of bounds.
# - Subset the array.
# - Get new transform for subset.
# - Create file with subset as data, default CRS, new transform, driver='GTiff'.
    
def get_combined_size(left: float, bottom: float, right: float, top: float) -> Tuple[int, int]:
    
    transform = get_transform(left, bottom, right, top)
    (row_max, row_min), (col_min, col_max) = rasterio.transform.rowcol(transform, (left, right), (bottom, top))
    return abs(row_max - row_min), abs(col_max - col_min) 
