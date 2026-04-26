import rasterio
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import rasterio
import numpy as np
from pathlib import Path

import rasterio
import numpy as np
from pathlib import Path

data_dir = Path("data/training_clipped")

for date_folder in sorted(data_dir.iterdir()):
    date = date_folder.name
    path = date_folder / f"{date}_Blue.tif"
    with rasterio.open(path) as src:
        band = src.read(1)
        # Compare left half vs right half
        left  = band[:, :band.shape[1]//2]
        right = band[:, band.shape[1]//2:]
    print(f"{date} | left  mean: {left.mean():.4f}  std: {left.std():.4f}")
    print(f"{date} | right mean: {right.mean():.4f}  std: {right.std():.4f}")
    print()