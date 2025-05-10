"""
batch_backfill_week.py
──────────────────────────────────────────────────────────────
Generate model predictions for the *past* week (168 hours) so
that Prediction-vs-Actual monitoring has data right away.
Outputs:
  • Parquet  → Resources/predictions/predictions_backfill.parquet
  • Feature group → citibike_predictions  (upsert)
"""
from pathlib import Path
import pandas as pd
import mlflow, hopsworks, yaml

CFG   = yaml.safe_load(open(Path("./configs/config.yaml")))
LAGS  = CFG["model"]["lags"]
HOURS_TO_BACKFILL = 7 * 24            # 168 h

LAG_COL_PATTERN = "lag_"
def _sort_lags(cols):                 # lag_1, lag_2, …
    return sorted(cols, key=lambda c: int(c.split("_")[1]))

def main():
    # ── connect to Hopsworks ───────────────────────────────────
    project = hopsworks.login(
        project       = CFG["project"]["name"],
        api_key_value = CFG["project"]["api_key"],
        host          = CFG["project"].get("host", "c.app.hopsworks.ai"),
    )
    fs = project.get_feature_store()
    mlflow.set_tracking_uri(CFG["mlflow"]["tracking_uri"])

    # ── load latest features ──────────────────────────────────
    feat_df = fs.get_feature_group("citibike_features", version=1).read()
    lag_cols = _sort_lags([c for c in feat_df.columns if c.startswith(LAG_COL_PATTERN)])

    # keep IDs as str (consistent with previous steps)
    feat_df["start_station_id"] = feat_df["start_station_id"].astype(str)

    # ── slice: last 168 h across *all* stations ───────────────
    latest_ts = feat_df["hour"].max()
    earliest  = latest_ts - pd.Timedelta(hours=HOURS_TO_BACKFILL - 1)

    week_df = feat_df[feat_df["hour"].between(earliest, latest_ts)].copy()

    # ── load production model ─────────────────────────────────
    model_dir = project.get_model_registry().get_model("citibike_best").download()
    model     = mlflow.pyfunc.load_model(model_dir)

    week_df["prediction"] = model.predict(week_df[lag_cols])

    # ── save parquet for Streamlit ────────────────────────────
    out_path = Path("predictions_backfill.parquet")
    week_df[["start_station_id", "hour", "prediction"]].to_parquet(out_path, index=False)
    project.get_dataset_api().upload(
        str(out_path), "Resources/predictions", overwrite=True
    )

    # ── upsert to predictions feature group ───────────────────
    pred_fg = fs.get_or_create_feature_group(
        name="citibike_predictions",
        version=1,
        primary_key=["start_station_id", "hour"],
        event_time="hour",
        time_travel_format="Hudi",
        description="24-hour Citi Bike forecasts (future + back-fill)",
    )
    pred_fg.insert(
        week_df[["start_station_id", "hour", "prediction"]],
        write_options={"wait_for_job": True},
    )
    print(
        f"✅  Back-filled {len(week_df)} station-hour rows "
        f"({earliest} → {latest_ts}). Monitoring should now show metrics."
    )

if __name__ == "__main__":
    main()
