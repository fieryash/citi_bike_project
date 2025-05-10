# app.py ─────────────────────────────────────────────────────────────
# Citi Bike Forecast & Monitoring dashboard (Streamlit)
# -------------------------------------------------------------------
import os
from pathlib import Path

import yaml
import hopsworks
import pandas as pd
import streamlit as st
import plotly.express as px

# ─────────────────── Load config ────────────────────
CFG = yaml.safe_load(open(Path("./configs/config.yaml")))

st.set_page_config(
    page_title="Citi Bike – Forecast & Monitoring",
    layout="wide",
    page_icon="🚲",
)
# ─────────────────── Load config ────────────────────
def load_config():
    config_path = Path("./configs/config.yaml")
    if config_path.exists():
        cfg = yaml.safe_load(open(config_path))
    else:
        cfg = {}

    # Override or fill from secrets if present
    if "project" not in cfg:
        cfg["project"] = {}

    project_cfg = cfg["project"]
    secrets_proj = st.secrets.get("project", {})

    project_cfg["name"] = secrets_proj.get("name")
    project_cfg["host"] = secrets_proj.get("host", "c.app.hopsworks.ai")
    project_cfg["api_key"] = secrets_proj.get("api_key")

    return cfg

CFG = load_config()
st.write("Using project:", CFG)

# -------------------------------------------------------------------
# 1️⃣  Sidebar navigation
# -------------------------------------------------------------------
st.sidebar.title("🚲 Citi Bike Dashboard")
PAGE = st.sidebar.radio("Go to", ["Forecast Dashboard", "Model Monitoring"])

# -------------------------------------------------------------------
# 2️⃣  Data loaders (cached)
# -------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="🔑  Logging in to Hopsworks…")
def _login():
    return hopsworks.login(
        project=CFG["project"]["name"],
        api_key_value=CFG["project"]["api_key"],
        host=CFG["project"].get("host", "c.app.hopsworks.ai"),
    )

@st.cache_data(ttl=3600, show_spinner="📦  Fetching feature groups…")
def _load_frames():
    project = _login()
    fs = project.get_feature_store()

    pred_df  = fs.get_feature_group("citibike_predictions", version=1).read()
    feats_df = fs.get_feature_group("citibike_features",   version=1).read()

    # Remove excluded station
    pred_df  = pred_df[~pred_df["start_station_id"].isin(["5788.13"])]
    feats_df = feats_df[~feats_df["start_station_id"].isin(["5788.13"])]

    for df in (pred_df, feats_df):
        df["hour"] = pd.to_datetime(df["hour"], utc=True)
        df["start_station_id"] = df["start_station_id"].astype(str)

    id2name_manual = {
        "6140.05": "W 21 St & 6 Ave",
        "5905.14": "University Pl & E 14 St",
        "5329.03": "West St & Chambers St",
    }

    return pred_df, feats_df, id2name_manual

pred_df, feats_df, ID2NAME = _load_frames()
ALL_STATIONS = sorted(pred_df["start_station_id"].unique())

def label(sid: str) -> str:
    return f"{sid} – {ID2NAME.get(sid, 'Unknown')}"

# -------------------------------------------------------------------
# 3️⃣  Forecast page
# -------------------------------------------------------------------
if PAGE == "Forecast Dashboard":
    st.title("📈  24-Hour Citi Bike Ride Forecast")

    selection = st.multiselect(
        "Select station(s)",
        options=ALL_STATIONS,
        default=[ALL_STATIONS[0]] if ALL_STATIONS else [],
        format_func=label,
    )
    if not selection:
        st.warning("Pick at least one station.")
        st.stop()

    sel_pred = pred_df[pred_df["start_station_id"].isin(selection)]

    total_pred  = int(sel_pred["prediction"].sum())
    mean_hourly = sel_pred.groupby("hour")["prediction"].sum().mean()

    kpi1, kpi2 = st.columns(2)
    kpi1.metric("Total Predicted Rides (24 h)", f"{total_pred:,}")
    kpi2.metric("Average Rides / Hour",         f"{mean_hourly:,.1f}")

    if len(selection) > 1:
        chart_series = sel_pred.groupby("hour")["prediction"].sum()
        st.line_chart(chart_series, height=350)
    else:
        st.line_chart(
            sel_pred.set_index("hour")["prediction"],
            height=350,
        )

    with st.expander("🔍  Raw predictions"):
        tbl = sel_pred.copy()
        tbl["start_station_name"] = tbl["start_station_id"].map(ID2NAME)
        st.dataframe(tbl.reset_index(drop=True), use_container_width=True)

# -------------------------------------------------------------------
# 4️⃣  Monitoring page
# -------------------------------------------------------------------
else:
    st.title("🩺  Model Monitoring – Prediction vs Actuals")

    joined = pred_df.merge(
        feats_df[["start_station_id", "hour", "rides"]],
        on=["start_station_id", "hour"],
        how="inner",
        suffixes=("_pred", "_act"),
    )

    selection = st.multiselect(
        "Select station(s) to monitor",
        options=ALL_STATIONS,
        default=[ALL_STATIONS[0]] if ALL_STATIONS else [],
        format_func=label,
    )
    if not selection:
        st.warning("Pick at least one station.")
        st.stop()

    jsel = joined[joined["start_station_id"].isin(selection)].copy()
    if jsel.empty:
        st.info("❔ No overlapping actuals yet for these predictions. "
                "Wait until rides appear in the feature group.")
        st.stop()

    jsel["error"]     = jsel["prediction"] - jsel["rides"]
    jsel["abs_error"] = jsel["error"].abs()
    jsel["ape"]       = jsel["abs_error"] / jsel["rides"].replace(0, pd.NA)

    mae  = round(jsel["abs_error"].mean(), 2)
    mape = jsel["ape"].mean()
    mape_str = "n/a"
    if pd.notna(mape):
        mape_str = f"{round(mape * 100, 2):.2f} %"

    col1, col2 = st.columns(2)
    col1.metric("MAE (joined horizon)", f"{mae:,.2f}")
    col2.metric("MAPE", mape_str)

    st.subheader("Actual vs Predicted Rides (Hourly Total)")

    # Aggregate across selected stations by hour
    agg_df = (
        jsel.groupby("hour")[["rides", "prediction"]]
        .sum()
        .reset_index()
        .sort_values("hour")
    )

    fig = px.line(
        agg_df,
        x="hour",
        y=["rides", "prediction"],
        title="Actual vs Predicted Hourly Rides (Aggregated)",
        labels={"value": "Number of Rides", "hour": "Time", "variable": "Type"},
        height=500,
    )

    fig.update_traces(mode="lines+markers")
    fig.update_layout(
        legend_title_text="Type",
        hovermode="x unified",
        template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly"
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍  Joined prediction / actual table"):
        jdisp = jsel.copy()
        jdisp["start_station_name"] = jdisp["start_station_id"].map(ID2NAME)
        st.dataframe(jdisp.reset_index(drop=True), use_container_width=True)
