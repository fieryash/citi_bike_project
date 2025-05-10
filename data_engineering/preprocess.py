"""
Placeholder for advanced cleaning (missing values, outlier removal, timezone normalization).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
import zipfile
import shutil

RAW_DIR = Path("tmp_raw")
CLEAN_DIR = RAW_DIR / "cleaned"
CLEAN_DIR.mkdir(exist_ok=True, parents=True)

# Columns to check for missing values
REQUIRED_COLS = [
    "ride_id", "rideable_type", "started_at", "ended_at",
    "start_station_name", "end_station_name", "start_lat", "start_lng", "end_lat", "end_lng"
]

# Outlier thresholds
MIN_TRIP_DURATION_SEC = 60  # 1 minute
MAX_TRIP_DURATION_SEC = 60 * 60  # 1 hour

def find_all_csvs_and_unzip(root_dir):
    csv_files = []
    for path in Path(root_dir).rglob("*.csv"):
        csv_files.append(path)
    for zip_path in Path(root_dir).rglob("*.zip"):
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.namelist():
                if member.endswith('.csv'):
                    extract_path = zip_path.parent / member
                    if not extract_path.exists():
                        zf.extract(member, zip_path.parent)
                    csv_files.append(extract_path)
    return csv_files

def main():
    csv_files = find_all_csvs_and_unzip(RAW_DIR)
    for csv_path in csv_files:
        # Skip macOS metadata and hidden files
        if csv_path.name.startswith("._") or "__MACOSX" in str(csv_path):
            print(f"Skipping macOS metadata or hidden file: {csv_path}")
            continue
        out_path = CLEAN_DIR / (csv_path.stem + "_cleaned.parquet")
        if out_path.exists():
            print(f"Skipping {csv_path}, cleaned file already exists.")
            continue
        print(f"Processing {csv_path}")
        try:
            df = pd.read_csv(csv_path, parse_dates=["started_at", "ended_at"], low_memory=False)
        except UnicodeDecodeError:
            print(f"Skipping non-CSV or corrupted file: {csv_path}")
            continue

        # Drop rows with missing required fields
        df = df.dropna(subset=REQUIRED_COLS)

        # Remove trips with negative or extreme durations
        df["trip_duration_sec"] = (df["ended_at"] - df["started_at"]).dt.total_seconds()
        df = df[(df["trip_duration_sec"] >= MIN_TRIP_DURATION_SEC) & (df["trip_duration_sec"] <= MAX_TRIP_DURATION_SEC)]

        # Normalize datetimes to UTC (assume input is local time, convert if needed)
        if df["started_at"].dt.tz is None:
            df["started_at"] = df["started_at"].dt.tz_localize("America/New_York", ambiguous='NaT').dt.tz_convert("UTC")
            df["ended_at"] = df["ended_at"].dt.tz_localize("America/New_York", ambiguous='NaT').dt.tz_convert("UTC")
        else:
            df["started_at"] = df["started_at"].dt.tz_convert("UTC")
            df["ended_at"] = df["ended_at"].dt.tz_convert("UTC")

        # Save cleaned file
        df.to_parquet(out_path, index=False)
        print(f"Saved cleaned data to {out_path}")

if __name__ == "__main__":
    main()