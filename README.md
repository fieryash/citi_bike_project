# 🚲 Citi Bike Trip Prediction System

A robust, production-grade pipeline for forecasting hourly ride counts for New York City’s Citi Bike program.  
This project demonstrates modern MLOps: automated data ingestion, feature engineering, model training, batch inference, monitoring, and CI/CD with Hopsworks, MLflow, and Streamlit.

---

## 🗂️ Project Structure

```
citi_bike_project/
├── configs/                # YAML config (project, model, data, MLflow, etc.)
├── data_engineering/       # Data download, extraction, and cleaning scripts
├── feature_engineering/    # Feature generation, lag creation, and validation
├── modeling/               # Model training, MLflow logging, Hopsworks registry
├── inference/              # Batch and backfill prediction jobs
├── monitoring/             # Streamlit-based monitoring dashboard
├── streamlit_app/          # Public UI for forecasts and monitoring
├── .github/workflows/      # CI/CD: scheduled and manual GitHub Actions
├── presentation/           # Project presentation (PPTX)
├── requirements.txt        # All Python dependencies
└── README.md               # This file
```

---

## 🚦 Pipeline Overview

1. **Data Engineering**
   - Downloads monthly and yearly Citi Bike trip data from S3.
   - Handles zipped and nested files, extracts all CSVs.
   - Cleans, deduplicates, and normalizes data (missing values, outliers, timezones).
   - Outputs cleaned Parquet files for each month.

2. **Feature Engineering**
   - Aggregates hourly ride counts for the top N busiest stations.
   - Generates lag features (e.g., previous 28 hours).
   - Saves features locally and ingests them into the Hopsworks Feature Store.

3. **Modeling & Training**
   - Loads features from Hopsworks.
   - Trains baseline, full-lag, and top-k LightGBM models.
   - Logs metrics and models to MLflow.
   - Registers and uploads the best model to the Hopsworks Model Registry.

4. **Batch Inference**
   - Loads the latest production model from Hopsworks.
   - Generates 24-hour ahead forecasts for each station.
   - Writes predictions to Parquet and Hopsworks Feature Group for monitoring.

5. **Monitoring & UI**
   - Streamlit dashboard for real-time forecast visualization and model monitoring.
   - Backfill job generates historical predictions for immediate monitoring.

6. **CI/CD**
   - GitHub Actions automate feature engineering and inference on schedule or push.
   - Secrets (API keys, MLflow URIs) are injected securely.

---

## 🚀 Quick Start

```bash
# 1. Environment setup
conda create -n citibike python=3.10 -y
conda activate citibike
pip install -r requirements.txt

# 2. Set secrets (locally or via CI/CD)
export HOPSWORKS_API_KEY="your-api-key"
export MLFLOW_TRACKING_URI="your-mlflow-uri"

# 3. Run the pipeline
python data_engineering/fetch_data.py           # Download & clean data
python data_engineering/preprocess.py           # Extract & preprocess all CSVs
python feature_engineering/generate_features.py # Feature engineering & store ingest
python modeling/train.py                        # Train & register models
python inference/batch_predict.py               # Batch inference for next 24h
python inference/backfill.py                    # Backfill for monitoring
streamlit run streamlit_app/app.py              # Launch dashboard
```

---

## 🛠️ Configuration

All tunable parameters (data, model, training, MLflow, etc.) are in [`configs/config.yaml`](configs/config.yaml):

- **Data:** top N stations, history window, S3 bucket, timezone
- **Model:** lags, LightGBM hyperparameters, top-k features
- **Training:** test split ratio, random seed
- **MLflow:** experiment/run names, tracking URI

---

## 🧩 Key Features

- **Robust Data Handling:** Recursively extracts and processes all Citi Bike data, including nested and zipped files.
- **Automated Feature Engineering:** Lag features, top-N station filtering, and time-based aggregations.
- **MLOps-Ready:** Full integration with Hopsworks Feature Store and Model Registry, MLflow experiment tracking, and GitHub Actions CI/CD.
- **Scalable Inference:** Batch and backfill jobs for real-time and historical predictions.
- **Monitoring:** Streamlit dashboard for live and historical model performance.
- **Reproducibility:** All configs, code, and artifacts are versioned and tracked.

---

## 🧪 Experiment Tracking & Model Registry

- **MLflow:** All experiments, metrics, and models are logged to MLflow (local or remote).
- **Hopsworks:** Features and models are versioned and stored in Hopsworks for production use.

---

## 🌐 Monitoring & Visualization

- **Streamlit App:** Real-time dashboard for forecasts, actuals, and model drift.
- **Backfill:** Ensures monitoring dashboard is populated from day one.

---

## 🤖 CI/CD

- **GitHub Actions:**  
  - `feature_engineering.yml`: Runs data fetch and feature generation daily.
  - `inference.yml`: Runs batch inference hourly.
- **Secrets:** All sensitive credentials are managed via GitHub repo secrets.

---

## 📁 Data Sources

- [Citi Bike S3 Bucket](https://s3.amazonaws.com/tripdata/index.html): All raw trip data (monthly, yearly, zipped).
- Data is automatically downloaded, extracted, and cleaned by the pipeline.

---

## 👤 Authors

- Ashtik Mahapatra

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 💡 Tips

- For large datasets, the pipeline processes files in batches to avoid memory issues.
- All IDs (e.g., station_id) are treated as strings for consistency.
- To change the number of top stations or lags, edit `configs/config.yaml`.
- For troubleshooting, check logs in each script and the Streamlit dashboard.

---

## 🏆 Example Results

- **Top 3 Stations:**  
  Run `feature_engineering/tests.py` to see the busiest start locations.
- **Model Performance:**  
  All MAE and other metrics are logged in MLflow and visible in the dashboard.

---

## 🙏 Acknowledgements

- [Citi Bike NYC](https://www.citibikenyc.com/system-data)
- [Hopsworks](https://www.hopsworks.ai/)
- [MLflow](https://mlflow.org/)
- [Streamlit](https://streamlit.io/)

---
