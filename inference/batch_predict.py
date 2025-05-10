"""
batch_predict.py
────────────────────────────────────────────────────────────────
1. Load the latest production model from Hopsworks Model Registry
2. Generate 24 h ahead forecasts for each start station
3. Write results:
      •  Parquet in Resources/predictions   (for Streamlit app)
      •  citibike_predictions Feature Group (Hudi time-travel)
"""

from pathlib import Path
import datetime as dt
import os

import pandas as pd
import numpy as np
import mlflow
import hopsworks
import yaml

# ─────────────────── Config ────────────────────
CFG = yaml.safe_load(open(Path("./configs/config.yaml")))

LAG_COL_PATTERN = "lag_"          # change if you renamed the lag features
PREDIC_HORIZON_H = 24             # hours to predict

# ─────────────────── Helpers ────────────────────
def _sort_lags(cols):
    """Sort lag column names numerically: lag_1, lag_2, …"""
    return sorted(cols, key=lambda c: int(c.split("_")[1]))

# ─────────────────── Main ────────────────────
def main():
    # 1️⃣  Connect to project
    project = hopsworks.login(
        project        = CFG["project"]["name"],
        api_key_value  = CFG["project"]["api_key"],
        host           = CFG["project"].get("host", "c.app.hopsworks.ai"),
    )
    fs = project.get_feature_store()
    mlflow.set_tracking_uri(CFG["mlflow"]["tracking_uri"])

    # 2️⃣  Read latest feature snapshot
    feat_fg = fs.get_feature_group("citibike_features", version=1)
    feat_df = feat_fg.read()                         # pandas
    lag_cols = _sort_lags([c for c in feat_df.columns if c.startswith(LAG_COL_PATTERN)])

    last_ts = feat_df["hour"].max()                  # tz-aware
    # last rows for ALL start stations (multi-row DF)
    current_df = feat_df[feat_df["hour"] == last_ts].copy()

    # 3️⃣  Load latest model
    mr = project.get_model_registry()
    model_meta = mr.get_model("citibike_best")       # latest version by default
    model_dir  = model_meta.download()               # local folder :contentReference[oaicite:0]{index=0}
    model      = mlflow.pyfunc.load_model(model_dir)

    # 4️⃣  Roll forward 24 h autoregressively
    preds_list = []

    for step in range(PREDIC_HORIZON_H):
        # predict
        X = current_df[lag_cols]
        preds = model.predict(X)

        # collect output rows
        out = current_df[["start_station_id"]].copy()
        out["hour"] = current_df["hour"] + pd.Timedelta(hours=step + 1)
        out["prediction"] = preds
        preds_list.append(out)

        # build next_df for next step
        next_df = current_df.copy()
        next_df["hour"] = out["hour"]               # advance 1 h
        # shift lags: lag_2 ← lag_1, lag_3 ← lag_2, …, lag_N ← lag_{N-1}
        next_df[lag_cols[1:]] = current_df[lag_cols[:-1]].values
        next_df[lag_cols[0]]  = preds               # new lag_1 = current prediction
        next_df["rides"]      = preds               # keep “rides” consistent
        current_df = next_df                        # loop

    pred_df = pd.concat(preds_list, ignore_index=True)

    # 5️⃣  Persist parquet (for Streamlit)
    parquet_path = Path("predictions.parquet")
    pred_df.to_parquet(parquet_path, index=False)
    project.get_dataset_api().upload(
        str(parquet_path), "Resources/predictions", overwrite=True
    )

    # 6️⃣  Insert into / create Feature Group
    pred_fg = fs.get_or_create_feature_group(
        name              = "citibike_predictions",
        version           = 1,
        primary_key       = ["start_station_id", "hour"],
        event_time        = "hour",
        description       = "24-hour ahead Citi Bike demand forecasts",
        time_travel_format= "Hudi",
    )
    pred_fg.insert(pred_df, write_options={"wait_for_job": True})
    print(f"✅  Stored {len(pred_df)} predictions up to {pred_df['hour'].max()}")

if __name__ == "__main__":
    main()
