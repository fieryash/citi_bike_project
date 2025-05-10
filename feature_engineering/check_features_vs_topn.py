import pandas as pd
from pathlib import Path
import yaml

# Load config
yaml_path = Path("configs/config.yaml")
with open(yaml_path) as f:
    CFG = yaml.safe_load(f)

# Load features
df_feat = pd.read_parquet("tmp_raw/citibike_features.parquet")
print(f"Features file has {len(df_feat)} rows.")

# Get top_n_stations from config
top_n = CFG["data"]["top_n_stations"]

# Count unique stations in features
unique_stations = df_feat["start_station_id"].nunique()
print(f"Features file contains {unique_stations} unique stations (top_n_stations in config: {top_n})")

# Print row counts per station
counts = df_feat["start_station_id"].value_counts()
print("Rows per station (top 10):\n", counts.head(10))
print("Rows per station (bottom 10):\n", counts.tail(10))

# Optionally, print all station IDs in features
#print("Station IDs in features:", df_feat["start_station_id"].unique())
