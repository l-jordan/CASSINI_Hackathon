# -------------------------------
# CALCULATING INDICES FROM SENTINEL-2 DATA
#---------------------------------

import rasterio 
import numpy as np
from pathlib import Path
from rasterio.enums import Resampling

# Data Configuration 
# ----------------------------------------
input_path = Path("./data/raw_training")
output_path = Path("./data/training")
output_path.mkdir(exist_ok=True)

# Saves index
# ----------------------------------------

def save_index(array, profile, out_path):
    profile.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array, 1)
        
# Resampling 20m to 10m
# ----------------------------------------
def resample_to_match(swir_path, blue_path):
    with rasterio.open(blue_path) as ref:
        ref_width     = ref.width
        ref_height    = ref.height

    with rasterio.open(swir_path) as src:
        with rasterio.open(blue_path) as ref:
            swir = src.read(1,out_shape=(ref.height, ref.width),
                            resampling=Resampling.bilinear).astype(np.float32) / 10_000
    return swir

# Index Calculations 
# ------------------------------------------
# Plastic greenhouse index (PGHI)
def ndvi_ind(NIR, R):
    ndvi = (NIR-R)/(NIR+R)
    return ndvi

# Enhanced vegetation index
def evi_ind(NIR, R, B):
    denom = (NIR + 6*R -7.5*B + 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(denom == 0, np.nan,
                        2.5 * (NIR - R) / denom).astype(np.float32)

# Plastic greenhouse index (PGHI)
def pghi_ind(B, SWIR2):
    pghi = B/SWIR2
    return pghi

# Colour Steel Buildings Index (CSBI)
def csbi_ind(SWIR1, SWIR2):
    csbi = SWIR2/SWIR1
    return csbi 

# -------------------------------------------
# Main
# -------------------------------------------

for date_folder in sorted(input_path.iterdir()):
    date = date_folder.name

    find = lambda band: date_folder / f"{date}_{band}.tif"

    # Locating the bands within file 'date_folder'    
    # B - Blue
    with rasterio.open(find("B02")) as src:
        B = src.read(1).astype(np.float32) / 10_000
        profile = src.profile.copy()
    # R - Red
    with rasterio.open(find("B03")) as src:
        R = src.read(1).astype(np.float32) / 10_000
        profile = src.profile.copy()
        
    # G - Green
    with rasterio.open(find("B04")) as src:
        G = src.read(1).astype(np.float32) / 10_000
        profile = src.profile.copy()
        
    # NIR 
    with rasterio.open(find("B08")) as src:
        NIR = src.read(1).astype(np.float32) / 10_000
        profile = src.profile.copy()
        
    # SWIR1 - Sampling to 10m grid
    SWIR1 = resample_to_match(find("B11"), find("B02"))
    
    # SWIR2
    SWIR2 = resample_to_match(find("B12"), find("B02"))
        
    # One output folder per date 
    out_dir = output_path/date
    out_dir.mkdir(parents=True, exist_ok=True)
    
    final_values = {
        "Red": R,
        "Green": G, 
        "Blue": B,
        "NDVI": ndvi_ind(NIR, R),
        "EVI": evi_ind(NIR, R, B),
        "PGHI":pghi_ind(B, SWIR2),
        "CSBI":pghi_ind(SWIR1, SWIR2)
    }
    
    print(final_values.items())
    # for index_name, array in indices.items():
    #     save_index(array, profile.copy(), out_dir / f"{date}_{index_name}.tif")
    #     print(f"{index_name}: {date}")
    