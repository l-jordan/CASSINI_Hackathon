import pandas as pd

## ORIGINAL VALUES ~ 3.1 million points
# exposed_slope       1467084
# water_bodies        1361862
# forest                98332
# solar_panels          18498
# green_fields          17601
# leafy_vegetation      13095
# exposed_rock          12788
# infrastructure         8053

df = pd.read_parquet(r"C:\\Users\\Laure\\Documents\\CASSINI_Hackathon\\CASSINI_Hackathon\\data\\training_data.parquet")

print(df.shape)        # (rows, cols) — expect one row per class
print(df["class"].value_counts())

counts = df["class"].value_counts()
min_count = counts.min() 
threshold = 5 * min_count  # ~3772

def smart_sample(g):
    n = len(g)
    if n > threshold:
        # Apply square-root scaling only to classes that are disproportionately large
        target = int(n ** 0.5 * 30)
        return g.sample(n=min(n, target), random_state=42)
    else:
        # Keep small classes untouched
        return g

df_balanced = (
    df.groupby("class")
    .apply(smart_sample)
    .reset_index(drop=True)
)

print(df_balanced["class"].value_counts())
print(df_balanced.shape)

df_balanced.to_parquet(
    r"C:\\Users\\Laure\\Documents\\CASSINI_Hackathon\\CASSINI_Hackathon\\data\\training_data_balanced.parquet",
    index=False
)