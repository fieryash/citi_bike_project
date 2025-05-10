import pandas as pd, numpy as np, hopsworks, pytz, yaml, os
from pathlib import Path
import datetime as dt

CFG = yaml.safe_load(open("./configs/config.yaml"))

FORCED_API_KEY = CFG["project"]["api_key"]
HOPS_HOST = CFG["project"].get("host", "c.app.hopsworks.ai")
LOCAL_TZ  = pytz.timezone(CFG["project"]["timezone"])      # America/New_York

def localise_start(ts_series: pd.Series) -> pd.Series:
    """Treat naïve datetimes as NYC local; convert others."""
    if ts_series.dt.tz is None:
        return ts_series.dt.tz_localize(LOCAL_TZ)
    return ts_series.dt.tz_convert(LOCAL_TZ)

def get_hourly_counts_batched():
    cleaned_dir = Path("tmp_raw/cleaned")
    files = list(cleaned_dir.glob("*_cleaned.parquet"))
    if not files:
        raise FileNotFoundError("No cleaned parquet files found in tmp_raw/cleaned.")

    # ---------- pass 1: find top-N stations ----------
    station_counts = {}
    for f in files:
        df = pd.read_parquet(f, columns=["start_station_id"])
        df["start_station_id"] = df["start_station_id"].astype(str)
        for sid, cnt in df["start_station_id"].value_counts().items():
            station_counts[sid] = station_counts.get(sid, 0) + cnt

    top_stations = (
        pd.Series(station_counts)
        .nlargest(CFG["data"]["top_n_stations"])
        .index
        .tolist()
    )

    # ---------- pass 2: hourly counts for top stations ----------
    hourly_parts = []
    for f in files:
        df = pd.read_parquet(f)
        df["start_station_id"] = df["start_station_id"].astype(str)
        df = df[df["start_station_id"].isin(top_stations)]

        df["started_at"] = localise_start(df["started_at"])
        df["hour"] = df["started_at"].dt.floor("H")

        grp = (
            df.groupby(["start_station_id", "hour"])
            .size()
            .reset_index(name="rides")
        )
        hourly_parts.append(grp)

    hourly = (
        pd.concat(hourly_parts, ignore_index=True)
        .groupby(["start_station_id", "hour"])
        .sum()
        .reset_index()
    )
    return hourly

def create_lag_features(df, lags=28):
    dfs = [df]
    for l in range(1, lags + 1):
        shifted = df.copy()
        shifted["hour"] += pd.Timedelta(hours=l)
        shifted = shifted.rename(columns={"rides": f"lag_{l}"})
        dfs.append(shifted)

    wide = dfs[0]
    for d in dfs[1:]:
        wide = wide.merge(d, on=["start_station_id", "hour"], how="left")

    lag_cols = [f"lag_{l}" for l in range(1, lags + 1)]
    wide[lag_cols] = wide[lag_cols].fillna(0)
    return wide

def is_new_data_present():
    """Return True if there is a raw CSV in tmp_raw for the current month that does not have a corresponding cleaned parquet in tmp_raw/cleaned."""
    today = dt.datetime.utcnow()
    y, m = today.year, today.month
    raw_dir = Path("tmp_raw")
    clean_dir = raw_dir / "cleaned"
    clean_dir.mkdir(exist_ok=True, parents=True)
    # Look for any CSV for this month
    pattern = f"{y}{str(m).zfill(2)}-citibike-tripdata*.csv"
    for csv in raw_dir.glob(pattern):
        parquet = clean_dir / (csv.stem + "_cleaned.parquet")
        if not parquet.exists():
            return True
    return False

def main():
    if not is_new_data_present():
        print("No new data to process for this month. Skipping feature generation.")
        return

    hourly = get_hourly_counts_batched()
    features = create_lag_features(hourly, CFG["model"]["lags"])
    features.to_parquet("tmp_raw/citibike_features.parquet", index=False)
    print("Feature engineering complete → tmp_raw/citibike_features.parquet")

    project = hopsworks.login(
        project       = CFG["project"]["name"],
        api_key_value = FORCED_API_KEY,
        host          = HOPS_HOST,
    )
    fs = project.get_feature_store()
    grp = fs.get_or_create_feature_group(
        name        = "citibike_features",
        version     = 1,
        description = "Lag features for hourly Citi Bike rides",
        primary_key = ["start_station_id", "hour"],
        event_time  = "hour",
        time_travel_format = "Hudi",
    )
    grp.insert(features, write_options={"wait_for_job": False})
    print("Feature-group ingestion job started on Hopsworks.")

if __name__ == "__main__":
    main()
