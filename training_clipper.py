"""
Clips pixel values in a raster to [min_val, max_val] and saves the result.
"""

import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm

input_path = Path("./data/training")
output_path = Path("./data/training_clipped")
output_path.mkdir(exist_ok=True)    
    
def clip_raster_values(input_path, output_path, min_val, max_val, nodata_val=None):
    with rasterio.open(input_path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = nodata_val or src.nodata

    # Mask nodata before clipping
    if nodata is not None:
        mask = data == nodata
        data = np.clip(data, min_val, max_val)
        data[mask] = nodata  # restore nodata pixels
    else:
        data = np.clip(data, min_val, max_val)

    profile.update(dtype=rasterio.float32)

    # Build output path: training_clipped/date/date_index_clipped.tif
    out_filename = input_path.stem + "_clipped.tif"
    out_file = output_path / out_filename

    with rasterio.open(out_file, "w", **profile) as dst:
        dst.write(data, 1)

# Processing data 
# -----------------------------------------------
def process_date(file_path, output_path, min_val, max_val):
    date    = file_path.parent.name          # extract date from parent folder name
    out_dir = output_path / date             # training_clipped/date/
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_raster_values(file_path, out_dir, min_val, max_val, nodata_val=None)
    
# Usage
if __name__ == '__main__':
    date_folders = sorted(input_path.iterdir())
    for date_folder in tqdm(date_folders, desc="Processing dates"):
        files = sorted(date_folder.glob("*.tif"))
    
        process_date(files[3], output_path, -1.0, 1.0)  # NDVI
        process_date(files[1], output_path, -1.0, 1.0)  # CBSI
        process_date(files[4], output_path,  0.0, 1.0)  # PGHI
