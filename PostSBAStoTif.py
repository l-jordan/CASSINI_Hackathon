import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path
from tqdm import tqdm

# =========================
# USER PATHS
# =========================
data_dir = Path(r"C:\Users\iamal\Documents\CassiniWell\Data\SBAS_Postprocess")

clean_cube_path  = data_dir / "disp_clean.dat"
velocity_path    = data_dir / "velocity.dat"
mask_path        = data_dir / "reliability_mask.dat"
aps_path         = data_dir / "aps_cube.dat"
meta_path        = data_dir / "postprocess_meta.npz"

out_dir = data_dir / "geotiff"

out_vel = out_dir / "velocity"
out_disp = out_dir / "displacement"
out_mask = out_dir / "mask"
out_aps  = out_dir / "aps"

for d in [out_vel, out_disp, out_mask, out_aps]:
    d.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD META
# =========================
meta = np.load(meta_path, allow_pickle=True)

print("Meta keys:", meta.files)

nrows = int(meta["nrows"])
ncols = int(meta["ncols"])
n_epochs = int(meta["n_epochs"])
t_days = meta["t_days"]

# =========================
# GEOREFERENCE (placeholder)
# =========================
ref_ifg = r"C:\Users\iamal\Documents\CassiniWell\Data\Interferograms_harmonised\Interferograms_harmonised\20160111_20160123.unw.tif"

with rasterio.open(ref_ifg) as src:
    transform = src.transform
    crs = src.crs

crs = "EPSG:4326"

# =========================
# LOAD DATA
# =========================
velocity = np.memmap(velocity_path, dtype="float32", mode="r",
                     shape=(nrows, ncols))

mask = np.memmap(mask_path, dtype="uint8", mode="r",
                 shape=(nrows, ncols))

disp = np.memmap(clean_cube_path, dtype="float32", mode="r",
                 shape=(n_epochs, nrows, ncols))

aps = np.memmap(aps_path, dtype="float32", mode="r",
                shape=(n_epochs, nrows, ncols))  # assumed same structure

# =========================
# CONSTANTS
# =========================
LAMBDA = 0.0555
phase_to_mm = (LAMBDA / (4 * np.pi)) * 1000
vel_scale = phase_to_mm * 365

valid_mask = mask.astype(bool)

# =========================
# VELOCITY EXPORT
# =========================
print("Exporting velocity GeoTIFF...")

vel = np.array(velocity)
vel[~valid_mask] = np.nan
vel_mm = vel * vel_scale

with rasterio.open(
    out_vel / "velocity.tif",
    "w",
    driver="GTiff",
    height=nrows,
    width=ncols,
    count=1,
    dtype="float32",
    crs=crs,
    transform=transform,
    nodata=np.nan,
    compress="LZW"
) as dst:
    dst.write(vel_mm.astype("float32"), 1)

# =========================
# MASK EXPORT
# =========================
print("Exporting reliability mask...")

with rasterio.open(
    out_mask / "reliability_mask.tif",
    "w",
    driver="GTiff",
    height=nrows,
    width=ncols,
    count=1,
    dtype="uint8",
    crs=crs,
    transform=transform,
    nodata=0,
    compress="LZW"
) as dst:
    dst.write(mask.astype("uint8"), 1)

# =========================
# DISPLACEMENT EXPORT
# =========================
print("Exporting displacement time series...")

for i in tqdm(range(n_epochs)):

    d = np.array(disp[i])
    d[~valid_mask] = np.nan
    d_mm = d * phase_to_mm

    epoch_name = str(meta["epochs"][i])

    with rasterio.open(
        out_disp / f"disp_{i:03d}_{epoch_name}.tif",
        "w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
        compress="LZW"
    ) as dst:
        dst.write(d_mm.astype("float32"), 1)

# =========================
# APS EXPORT
# =========================
print("Exporting APS time series...")

for i in tqdm(range(n_epochs)):

    a = np.array(aps[i])
    a[~valid_mask] = np.nan

    epoch_name = str(meta["epochs"][i])

    with rasterio.open(
        out_aps / f"aps_{i:03d}_{epoch_name}.tif",
        "w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
        compress="LZW"
    ) as dst:
        dst.write(a.astype("float32"), 1)

print("Done. All GeoTIFFs written to:", out_dir)