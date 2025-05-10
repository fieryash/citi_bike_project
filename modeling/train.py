"""
train.py ― Citi Bike hourly-rides forecasting
=============================================
• Loads engineered features from the Feature Store
• Trains baseline, full-lags, and top-k LightGBM models
• Logs runs/metrics to MLflow, registers best model
• Uploads best model artefacts to the Hopsworks Model Registry
"""

import os
import shutil
from pathlib import Path
import datetime as dt

import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error

import hopsworks

from utils import CFG   # your yaml-backed config object

# ─────────────────────────── helpers ─────────────────────────── #

def load_features(project) -> pd.DataFrame:
    """Read the ‘citibike_features’ FG (v1) and return a DataFrame."""
    fs = project.get_feature_store()
    fg = fs.get_feature_group("citibike_features", version=1)
    return fg.read()                     # Spark → Pandas automatically


def train_model(df: pd.DataFrame, feature_cols: list[str]):
    """Train LightGBM and return (model, MAE on test split)."""
    X = df[feature_cols]
    y = df["rides"]

    split_idx = int(len(df) * (1 - CFG["training"]["test_ratio"]))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = lgb.LGBMRegressor(**CFG["model"]["lgb_params"])
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    return model, mae


def is_new_features_present():
    """Return True if features for the current month are present and updated in tmp_raw/citibike_features.parquet."""
    features_path = Path("tmp_raw/citibike_features.parquet")
    if not features_path.exists():
        return False
    # Check if features file was modified in the last 2 hours (to match hourly schedule)
    mtime = dt.datetime.fromtimestamp(features_path.stat().st_mtime)
    now = dt.datetime.utcnow()
    return (now - mtime).total_seconds() < 2 * 3600


# ─────────────────────────── main entry ─────────────────────────── #

def main() -> None:
    if not is_new_features_present():
        print("No new features to train on. Skipping model training.")
        return

    # 1️⃣  ── Connect to Hopsworks project
    project = hopsworks.login(
        project     = CFG["project"]["name"],
        api_key_value = CFG["project"]["api_key"],
        host        = CFG["project"].get("host", "c.app.hopsworks.ai"),
    )

    df = load_features(project).sort_values("hour")

    # 2️⃣  ── Start an MLflow run
    mlflow.set_experiment(CFG["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="citibike_training") as run:
        models: dict[str, tuple[lgb.LGBMRegressor, float] | None] = {}

        # Baseline (yesterday same hour)
        df["baseline"] = df["lag_24"]
        mae_base = mean_absolute_error(df["rides"], df["baseline"])
        mlflow.log_metric("baseline_mae", mae_base)
        models["baseline"] = None   # no artefact to store

        # Full-lag model
        full_feats = [c for c in df.columns if c.startswith("lag_")]
        m_full, mae_full = train_model(df, full_feats)
        mlflow.log_metric("full_mae", mae_full)
        mlflow.sklearn.log_model(m_full, "full_lag_model")
        models["full"] = (m_full, mae_full)

        # Top-k model (feature-importance pruning)
        importances = m_full.booster_.feature_importance(importance_type="gain")
        imp_series = pd.Series(importances, index=full_feats).sort_values(ascending=False)
        topk_feats = imp_series.head(CFG["model"]["features_top_k"]).index.tolist()

        m_top, mae_top = train_model(df, topk_feats)
        mlflow.log_metric("topk_mae", mae_top)
        mlflow.sklearn.log_model(m_top, "topk_model")
        models["topk"] = (m_top, mae_top)

        # 3️⃣  ── Pick best model (lowest MAE)
        best_name, (best_model, best_mae) = min(
            ((n, v) for n, v in models.items() if v is not None),
            key=lambda x: x[1][1],
        )

        # Register & promote in MLflow
        mlflow_model_name = "citibike_best"
        model_uri = f"runs:/{run.info.run_id}/{best_name}_model"
        mlflow.register_model(model_uri, mlflow_model_name)

        client = MlflowClient()
        version = client.get_latest_versions(mlflow_model_name, stages=["None"])[0].version
        client.transition_model_version_stage(
            name   = mlflow_model_name,
            version= version,
            stage  = "Production",
        )

        # 4️⃣  ── Export artefacts for Hopsworks upload
        local_model_dir = Path("tmp_raw/best_model").resolve()
        if local_model_dir.exists():
            shutil.rmtree(local_model_dir)
        mlflow.sklearn.save_model(best_model, str(local_model_dir))

    # 5️⃣  ── Upload to Hopsworks Model Registry
    mr = project.get_model_registry()
    skl_meta = mr.sklearn.create_model(
        name        = "citibike_best",
        metrics     = {"mae": best_mae},
        description = "Best LightGBM model for hourly Citi Bike rides prediction",
        input_example = df.iloc[[0]],
    )
    skl_meta.save(str(local_model_dir))     # uploads artefacts
    print(
        f"✅  Uploaded model version {skl_meta.version} (MAE = {best_mae:.4f}) to "
        f"Hopsworks Model Registry and promoted v{version} in MLflow."
    )


if __name__ == "__main__":
    main()
