####################################
# Inference script
## Classifies to get output images
####################################

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import rasterio
import joblib
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class MLP_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP_Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes)
        )
    def forward(self, x):
        return self.network(x)

##############################################
# Configuration
##############################################
data_dir    = Path("data/training_clipped")
model_path  = "MLP_files/model.pth"
scalar_path = "MLP_files/scaler.pkl"
data_path   = "data/training_data_balanced.parquet"
output_tif  = f"data/output/classified_{datetime.now().strftime('%H%M%S')}.tif"
tile_size   = 1024
no_data_val = -9999

band_map = {
    "Blue":         "Blue",
    "Green":        "Green",
    "NDVI_clipped": "NDVI_clipped",
    "PGHI_clipped": "PGHI_clipped",
    "Red":          "Red",
}

##############################################
# Build band file list dynamically
##############################################
band_files = []
for band_name, file_suffix in band_map.items():
    for date_folder in sorted(data_dir.iterdir()):
        date = date_folder.name
        band_files.append(date_folder / f"{date}_{file_suffix}.tif")

with rasterio.open(band_files[0]) as src:
    print("nodata value:", src.nodata)
    band = src.read(1)
    print("zeros:", (band == 0).sum())
    print("min value:", band.min())
##############################################
# Load class names, model, scaler
##############################################
df          = pd.read_parquet(data_path)
class_names = sorted(df["class"].unique().tolist())
train_min   = df.iloc[:, 3:].min().values.astype(np.float32)
train_max   = df.iloc[:, 3:].max().values.astype(np.float32)

model = MLP_Classifier(input_size=25, hidden_size=256, num_classes=8)
model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
model.eval()
print("Model loaded.")

scaler = joblib.load(scalar_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
print(f"Running on: {device}")

##############################################
# Get image dimensions from first band
##############################################
with rasterio.open(band_files[0]) as src:
    profile       = src.profile.copy()
    height, width = src.height, src.width
    nodata        = src.nodata or no_data_val

##############################################
# Output: single-band int8 classification raster
##############################################
profile.update(count=1, dtype=rasterio.int8, nodata=-1)
Path("data/output").mkdir(parents=True, exist_ok=True)

total_tiles = ((height + tile_size - 1) // tile_size) * \
              ((width  + tile_size - 1) // tile_size)
tile_num = 0

with rasterio.open(output_tif, "w", **profile) as dst:
    for row_off in range(0, height, tile_size):
        for col_off in range(0, width, tile_size):
            tile_h = min(tile_size, height - row_off)
            tile_w = min(tile_size, width  - col_off)
            tile_num += 1
            print(f"  Tile {tile_num}/{total_tiles}", end="\r")

            window = rasterio.windows.Window(col_off, row_off, tile_w, tile_h)

            # Read all 25 bands for this tile
            tile_pixels = np.zeros((tile_h * tile_w, 25), dtype=np.float32)
            for band_idx, band_path in enumerate(band_files):
                with rasterio.open(band_path) as src:
                    band = src.read(1, window=window).astype(np.float32)
                band = np.clip(band, train_min[band_idx], train_max[band_idx])
                tile_pixels[:, band_idx] = band.flatten()

            # Mask outside AOI
            valid_mask = (tile_pixels > 0.001).any(axis=1)

            # Normalise + predict valid pixels only
            tile_scaled = scaler.transform(tile_pixels)
            preds = np.full(tile_h * tile_w, -1, dtype=np.int8)

            if valid_mask.any():
                with torch.no_grad():
                    logits = model(torch.FloatTensor(tile_scaled[valid_mask]).to(device))
                    preds[valid_mask] = torch.argmax(logits, dim=1).cpu().numpy()

            dst.write(
                preds.reshape(tile_h, tile_w),
                1,
                window=window
            )
        dst.update_tags(**{str(i): name for i, name in enumerate(class_names)})

print(f"\nDone! Saved → {output_tif}")

##############################################
# Class breakdown
##############################################
print("\nClassification summary:")
with rasterio.open(output_tif) as src:
    result = src.read(1)
for i, name in enumerate(class_names):
    count = (result == i).sum()
    print(f"  {name:<20s}: {count:,} pixels  ({count / result.size * 100:.1f}%)")