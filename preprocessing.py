from concurrent.futures import ProcessPoolExecutor
import rasterio 
import numpy as np
from pathlib import Path
from rasterio.enums import Resampling
from tqdm import tqdm

# Data Configuration 
# ----------------------------------------
input_path = Path("./data/raw_training")
output_path = Path("./data/training")
output_path.mkdir(exist_ok=True)

scale = 10000

# Resampling 20m to 10m
# -----------------------------------------
def resample_to_match(swir_path, blue_path):
    with rasterio.open(blue_path) as ref:
        ref_width     = ref.width
        ref_height    = ref.height

    with rasterio.open(swir_path) as src:
        with rasterio.open(blue_path) as ref:
            swir = src.read(1,out_shape=(ref.height, ref.width),
                            resampling=Resampling.bilinear).astype(np.float32) / 10_000
    return swir

# Loading bands
# ----------------------------------------
def loading_bands(date_folder, date):
    find = lambda band: date_folder / f"{date}_{band}.tif"

    # Locating the bands within file 'date_folder'    
    with rasterio.open(find("B02")) as b02, \
        rasterio.open(find("B03")) as b03, \
        rasterio.open(find("B04")) as b04, \
        rasterio.open(find("B08")) as b08: 
            
        profile = b02.profile.copy() # Used for saving files 
            
        bands_10m = np.stack([
            b02.read(1),
            b03.read(1),
            b04.read(1),
            b08.read(1),
        ]).astype(np.float32)/scale
        
    # SWIR1 - Sampling to 10m grid
    SWIR1 = resample_to_match(find("B11"), find("B02"))
    
    # SWIR2
    SWIR2 = resample_to_match(find("B12"), find("B02"))
    
    B, G, R, NIR = bands_10m[0], bands_10m[1], bands_10m[2], bands_10m[3]
    
    return B, G, R, NIR, SWIR1, SWIR2, profile

# Saves index
# ----------------------------------------

def save_index(array, profile, out_path):
    profile.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array, 1)
        

# Index Calculations 
# ------------------------------------------
def export_indices(B, G, R, NIR, SWIR1, SWIR2):
    with np.errstate(invalid="ignore", divide="ignore"):

        ndvi = (NIR-R)/(NIR+R) # Normalised difference vegetation index
        pghi = B/SWIR2
        csbi = SWIR2/SWIR1

        return {
            "Red":   R,
            "Green": G,
            "Blue":  B,
            "NDVI":  ndvi.astype(np.float32),
            "PGHI":  pghi.astype(np.float32),
            "CSBI":  csbi.astype(np.float32),
        }
        

# Saving file function 
# ------------------------------------------
def save_all(indices, profile, out_dir, date):
    profile.update(dtype="float32", count=1, nodata=np.nan)
    for name, array in indices.items():
        out_path = out_dir / f"{date}_{name}.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(np.where(np.isfinite(array), array, np.nan).astype(np.float32), 1)

# Loading and saving files
# ------------------------------------------
def process_date(date_folder):
    date    = date_folder.name
    out_dir = output_path / date
    out_dir.mkdir(parents=True, exist_ok=True)

    B, G, R, NIR, SWIR1, SWIR2, profile = loading_bands(date_folder, date)
    indices = export_indices(B, G, R, NIR, SWIR1, SWIR2)
    save_all(indices, profile, out_dir, date)
    
if __name__ == '__main__':
    date_folders = sorted(input_path.iterdir())
    
    for date_folder in tqdm(date_folders, desc="Processing dates"):
        process_date(date_folder)