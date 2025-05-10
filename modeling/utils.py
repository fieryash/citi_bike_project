import mlflow, os, yaml
from pathlib import Path
CFG = yaml.safe_load(open("./configs/config.yaml"))
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
