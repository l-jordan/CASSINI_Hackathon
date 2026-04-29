import rasterio
import geopandas as gpd
from rasterio.features import rasterize
import numpy as np

labels = gpd.read_file('data/labels/water_bodies.geojson')

# Load reference raster to get transform, CRS and shape
with rasterio.open("data/raw_training/20210214/20210214_B02.tif") as ref:
    transform = ref.transform
    crs       = ref.crs
    shape     = (ref.height, ref.width)

# Reproject to match raster if needed
water_gdf = labels.to_crs(crs)

# Rasterize
burned = rasterize(
    [(geom, 1) for geom in water_gdf.geometry],
    out_shape=shape,
    transform=transform,
    fill=0,
    dtype='uint8'
)

pixel_count = burned.sum()
print(f"Pixels: {pixel_count:,}")
