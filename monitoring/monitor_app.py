import streamlit as st, hopsworks, pandas as pd, os
st.set_page_config(page_title="Citi Bike Model Monitoring", layout="wide")

project = hopsworks.login()
fs = project.get_feature_store()
pred_grp = fs.get_feature_group("citibike_predictions", version=1)
pred_df = pred_grp.read()

st.title("Citi Bike Rides – Prediction vs Actuals")
latest = pred_df.sort_values("hour").tail(24)
st.line_chart(latest.pivot(index="hour", columns="start_station_id", values="prediction"))
