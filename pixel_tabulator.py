# -- coding: utf-8 --
"""
Pixel-level feature table from Sentinel TIF bands/indices inside GeoJSON polygons.
Output: 1 row per pixel, columns = band_date values, plus a 'class' label column.
"""

# -- coding: utf-8 --
"""
Pixel-level feature table from Sentinel TIF bands/indices inside GeoJSON polygons.
Output: 1 row per pixel, columns = band_date values, plus a 'class' label column.
"""

import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.crs import CRS
from pathlib import Path
from tqdm import tqdm

# === CONFIG ===

geojson_folder  = r"C:\\Users\\Laure\\Documents\\CASSINI_Hackathon\\CASSINI_Hackathon\\data\\labels"
training_folder = r"C:\\Users\\Laure\\Documents\\CASSINI_Hackathon\\CASSINI_Hackathon\\data\\training_clipped"
output_parquet  = r"C:\\Users\\Laure\\Documents\\CASSINI_Hackathon\\CASSINI_Hackathon\\data\\training_data.parquet"

SKIP_EXTENSIONS = {".xml", ".ovr", ".aux", ".tfw", ".prj"}

# === LOAD CLASS POLYGONS ===

def load_class_geodataframes(geojson_folder: str) -> dict[str, gpd.GeoDataFrame]:
    classes = {}
    for path in Path(geojson_folder).glob("*.geojson"):
        class_name = path.stem
        gdf = gpd.read_file(path)
        if gdf.empty:
            print(f"⚠️  Empty GeoJSON: {path.name}")
            continue
        classes[class_name] = gdf
        print(f"  ✓ Loaded class '{class_name}' — {len(gdf)} polygon(s)")
    return classes

# === INDEX TIF FILES ===

def index_date_folders(training_folder: str) -> dict[str, dict[str, str]]:
    date_index = {}
    date_pattern = re.compile(r"^\d{8}$")

    for entry in sorted(os.scandir(training_folder), key=lambda e: e.name):
        if not entry.is_dir() or not date_pattern.match(entry.name):
            continue
        date_str = entry.name
        bands = {}
        for fpath in sorted(Path(entry.path).iterdir()):
            if fpath.suffix.lower() in SKIP_EXTENSIONS:
                continue
            if fpath.suffix.lower() != ".tif":
                continue
            stem = fpath.stem
            band_name = stem[len(date_str) + 1:] if stem.startswith(date_str + "_") else stem
            bands[band_name] = str(fpath)
        if bands:
            date_index[date_str] = bands
    return date_index

# === MASK BUILDING ===

def build_class_masks(
    class_gdfs: dict[str, gpd.GeoDataFrame],
    ref_transform,
    ref_shape: tuple[int, int],
    ref_crs: CRS,
) -> dict[str, np.ndarray]:
    masks = {}
    for class_name, gdf in class_gdfs.items():
        if gdf.crs != ref_crs:
            gdf = gdf.to_crs(ref_crs)
        geometries = list(gdf.geometry)
        if not geometries:
            continue
        mask = ~geometry_mask(
            geometries,
            out_shape=ref_shape,
            transform=ref_transform,
            all_touched=False,
        )
        if mask.any():
            masks[class_name] = mask
            print(f"    ✓ '{class_name}' — {mask.sum()} pixels")
        else:
            print(f"    ⚠️  No pixels inside '{class_name}' (check CRS/extent)")
    return masks

# === PIXEL EXTRACTION ===

def extract_pixels_for_date(
    date_str: str,
    band_paths: dict[str, str],
    class_masks: dict[str, np.ndarray],
    ref_shape: tuple[int, int],
) -> pd.DataFrame:
    """
    For one date, extract per-pixel band values for all masked pixels.
    Returns a DataFrame with columns [class, band_date, ...] — one row per pixel.
    """
    # Build a combined pixel index: for each class, get the flat indices of masked pixels
    # We'll stack all bands into a dict of {col_name: flat_array}
    band_arrays: dict[str, np.ndarray] = {}

    for band_name, fpath in band_paths.items():
        with rasterio.open(fpath) as ds:
            if (ds.height, ds.width) != ref_shape:
                from rasterio.enums import Resampling
                data = ds.read(1, out_shape=ref_shape,
                               resampling=Resampling.bilinear).astype(np.float32)
            else:
                data = ds.read(1).astype(np.float32)
            if ds.nodata is not None:
                data = np.where(data == ds.nodata, np.nan, data)
        band_arrays[f"{band_name}_{date_str}"] = data.ravel()  # flatten to 1D

    # For each class, extract the rows corresponding to its masked pixels
    frames = []
    for class_name, mask in class_masks.items():
        flat_idx = np.where(mask.ravel())[0]  # indices of pixels inside this class
        chunk = {col: arr[flat_idx] for col, arr in band_arrays.items()}
        chunk["class"] = class_name
        chunk["pixel_idx"] = flat_idx          # useful for spatial debugging
        frames.append(pd.DataFrame(chunk))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# === MAIN ===

if __name__ == "__main__":
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)

    print("Loading class GeoJSONs...")
    class_gdfs = load_class_geodataframes(geojson_folder)
    if not class_gdfs:
        raise RuntimeError("No GeoJSON files found. Check 'geojson_folder'.")

    print(f"\nIndexing date folders in: {training_folder}")
    date_index = index_date_folders(training_folder)
    print(f"Found {len(date_index)} date folder(s): {sorted(date_index)}\n")

    all_frames: list[pd.DataFrame] = []

    for date_str in tqdm(sorted(date_index), desc="Processing dates"):
        band_paths = date_index[date_str]
        tqdm.write(f"  {date_str} — {len(band_paths)} band(s): {sorted(band_paths)}")

        ref_path = next(iter(band_paths.values()))
        with rasterio.open(ref_path) as ref_ds:
            ref_transform = ref_ds.transform
            ref_shape     = (ref_ds.height, ref_ds.width)
            ref_crs       = ref_ds.crs

        class_masks = build_class_masks(class_gdfs, ref_transform, ref_shape, ref_crs)
        if not class_masks:
            tqdm.write(f"  No valid masks for {date_str} — skipping")
            continue

        date_df = extract_pixels_for_date(date_str, band_paths, class_masks, ref_shape)
        if not date_df.empty:
            all_frames.append(date_df)

    if all_frames:
        # Merge all dates on pixel_idx + class, so each pixel is one row with all band_date columns
        from functools import reduce
        df = reduce(
            lambda left, right: pd.merge(left, right, on=["class", "pixel_idx"], how="outer"),
            all_frames,
        )
        # Sort columns: class and pixel_idx first, then band_date columns
        band_cols = sorted([c for c in df.columns if c not in ("class", "pixel_idx")])
        df = df[["class", "pixel_idx"] + band_cols]
        df = df.sort_values(["class", "pixel_idx"]).reset_index(drop=True)

        df.to_parquet(output_parquet, index=False)
        print(f"\nDone — {len(df)} pixel rows, {len(df.columns)} columns")
        print(f"   Output: {output_parquet}")
        print(f"   Classes: {sorted(df['class'].unique().tolist())}")
        print(f"   Class counts:\n{df['class'].value_counts()}")
        print(f"   Sample columns: {list(df.columns[:8])} ...")
    else:
        print("No data extracted. Check folder paths, CRS, and polygon extents.")