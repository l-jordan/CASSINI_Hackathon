import pandas as pd

df = pd.read_parquet(r"C:\\Users\\Laure\\Documents\\CASSINI_Hackathon\\CASSINI_Hackathon\\data\\training_data.parquet")

print(df.shape)        # (rows, cols) — expect one row per class
print(df.head())       # first 5 rows
print(df.dtypes)       # confirm float32/64, no accidental strings
print(df.isnull().sum())  # check how many NaNs per column
print(df.iloc[1])