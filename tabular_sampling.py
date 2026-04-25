# -------------------------------
# CREATING TABULAR INPUT USING PYTHON 
# Rasterises the polygons from 'labels' and then samples your raster pixels into a flat table
#---------------------------------

import rasterio 
import pandas as pd
from pathlib import Path
from rasterio.mask import mask as rio_mask
import geopandas as gpd
import numpy as np 

# Configuration 
# ----------------------------------------
training_path = Path(".data/training")
labels_path = Path(".data/training")
output_file = Path(".data/training_data.parquet")

class_map = {
    "buildings.geojson": "building", 
    "exposed_rock.geojson": "rock/dirt",
    "exposed_slope.geojson": "slope/terrain",
    "forest.geojson": "forest",
    "leafy_vegetation.geojson": "leafy vegetation/fields",
    "solar_panels.geojson": "solar panels",
    "vegetation_cover.geojson": "vegetation/fields",
    "water_bodies.geojson": "water",
}