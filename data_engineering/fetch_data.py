"""
Download monthly Citi Bike tripdata from the official site and upload raw parquet to Hopsworks.

Usage:
python data_engineering/fetch_data.py --year 2024 --month 1
"""
import argparse, os, requests, zipfile, io, pandas as pd, hopsworks, pytz, datetime as dt
from pathlib import Path
from yaml import safe_load

CFG = safe_load(open("./configs/config.yaml"))

# Force API key for Hopsworks
import os
from dotenv import load_dotenv
load_dotenv()

FORCED_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPS_HOST = CFG["project"].get("host", "c.app.hopsworks.ai")

def download_file(url: str, out_dir: Path) -> Path:
    print("Downloading", url)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    filename = url.split("/")[-1]
    out_path = out_dir / filename
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path

def extract_zip(zip_path: Path, out_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)
    # Return first CSV found
    for f in out_dir.glob("*.csv"):
        return f
    return None

def process_and_upload(csv_path: Path, out_dir: Path, parquet_name: str):
    df = pd.read_csv(csv_path, parse_dates=["started_at", "ended_at"], low_memory=False)
    parquet_path = out_dir / parquet_name
    df.to_parquet(parquet_path, index=False)
    upload_to_hopsworks(parquet_path)

def upload_to_hopsworks(local_path:Path):
    project = hopsworks.login(project=CFG["project"]["name"], api_key_value=FORCED_API_KEY, host=HOPS_HOST)
    fs = project.get_feature_store()
    dataset_api = project.get_dataset_api()
    dataset_api.upload(local_path.as_posix(), "Resources/raw", overwrite=True)
    print("Uploaded to Hopsworks")

def month_year_iter(start_year, start_month, end_year, end_month):
    ym_start = 12*start_year + start_month - 1
    ym_end = 12*end_year + end_month - 1
    for ym in range(ym_start, ym_end+1):
        y, m = divmod(ym, 12)
        yield y, m+1

def is_new_data_available(latest_year, latest_month, out_dir):
    """Check if the latest month's data is already present in tmp_raw."""
    # Check for both .csv and .csv.zip
    csv_name = f"{latest_year}{str(latest_month).zfill(2)}-citibike-tripdata.csv"
    zip_name = f"{latest_year}{str(latest_month).zfill(2)}-citibike-tripdata.csv.zip"
    zip2_name = f"{latest_year}{str(latest_month).zfill(2)}-citibike-tripdata.zip"
    for fname in [csv_name, zip_name, zip2_name]:
        if (out_dir / fname).exists():
            return False  # Already present
    return True

def url_exists(url):
    try:
        r = requests.head(url, timeout=10)
        return r.status_code == 200
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=False)
    ap.add_argument("--month", type=int, required=False)
    ap.add_argument("--start_year", type=int, required=False)
    ap.add_argument("--start_month", type=int, required=False)
    ap.add_argument("--end_year", type=int, required=False)
    ap.add_argument("--end_month", type=int, required=False)
    ap.add_argument("--hourly", action="store_true", help="Check for new data and only fetch/process if new data is available.")
    args = ap.parse_args()
    out_dir = Path("tmp_raw"); out_dir.mkdir(exist_ok=True)

    # If --hourly, only fetch/process the latest month if new data is available
    if args.hourly:
        today = dt.datetime.utcnow()
        latest_year = today.year
        latest_month = today.month
        if not is_new_data_available(latest_year, latest_month, out_dir):
            print(f"No new data for {latest_year}-{latest_month:02d}. Skipping fetch.")
            return
        print(f"New data detected for {latest_year}-{latest_month:02d}. Fetching...")
        args.year = latest_year
        args.month = latest_month
        # Only fetch the latest month and return after processing
        url1 = f"https://s3.amazonaws.com/tripdata/{args.year}{str(args.month).zfill(2)}-citibike-tripdata.csv.zip"
        url2 = f"https://s3.amazonaws.com/tripdata/{args.year}{str(args.month).zfill(2)}-citibike-tripdata.zip"
        url = url1 if url_exists(url1) else url2 if url_exists(url2) else None
        if url:
            zip_path = download_file(url, out_dir)
            csv_path = extract_zip(zip_path, out_dir)
            if csv_path:
                process_and_upload(csv_path, out_dir, f"clean_{args.year}_{str(args.month).zfill(2)}.parquet")
        else:
            print(f"No data file found for {args.year}-{args.month:02d}")
        return

    # If no arguments, default to current year and previous two years
    if not any([args.year, args.month, args.start_year, args.start_month, args.end_year, args.end_month]):
        today = dt.datetime.now()
        args.start_year = today.year - 2
        args.start_month = 1
        args.end_year = today.year
        args.end_month = today.month

    # Download yearly files if available, else download monthly files
    if args.start_year and args.start_month and args.end_year and args.end_month:
        for year in range(args.start_year, args.end_year+1):
            # If full year is in range and yearly file exists, use it
            is_full_year = (year > args.start_year and year < args.end_year) or \
                (year == args.start_year and args.start_month == 1 and (year < args.end_year or args.end_month == 12)) or \
                (year == args.end_year and args.end_month == 12 and (year > args.start_year or args.start_month == 1))
            yearly_url = f"https://s3.amazonaws.com/tripdata/{year}-citibike-tripdata.zip"
            if is_full_year and url_exists(yearly_url):
                zip_path = download_file(yearly_url, out_dir)
                csv_path = extract_zip(zip_path, out_dir)
                if csv_path:
                    process_and_upload(csv_path, out_dir, f"clean_{year}.parquet")
                continue
            # Otherwise, download each month
            start_m = args.start_month if year == args.start_year else 1
            end_m = args.end_month if year == args.end_year else 12
            for month in range(start_m, end_m+1):
                # Try .csv.zip first, then .zip
                url1 = f"https://s3.amazonaws.com/tripdata/{year}{str(month).zfill(2)}-citibike-tripdata.csv.zip"
                url2 = f"https://s3.amazonaws.com/tripdata/{year}{str(month).zfill(2)}-citibike-tripdata.zip"
                url = url1 if url_exists(url1) else url2 if url_exists(url2) else None
                if url:
                    zip_path = download_file(url, out_dir)
                    csv_path = extract_zip(zip_path, out_dir)
                    if csv_path:
                        process_and_upload(csv_path, out_dir, f"clean_{year}_{str(month).zfill(2)}.parquet")
    elif args.year and args.month:
        # Try .csv.zip first, then .zip
        url1 = f"https://s3.amazonaws.com/tripdata/{args.year}{str(args.month).zfill(2)}-citibike-tripdata.csv.zip"
        url2 = f"https://s3.amazonaws.com/tripdata/{args.year}{str(args.month).zfill(2)}-citibike-tripdata.zip"
        url = url1 if url_exists(url1) else url2 if url_exists(url2) else None
        if url:
            zip_path = download_file(url, out_dir)
            csv_path = extract_zip(zip_path, out_dir)
            if csv_path:
                process_and_upload(csv_path, out_dir, f"clean_{args.year}_{str(args.month).zfill(2)}.parquet")
        else:
            raise ValueError(f"No data file found for {args.year}-{args.month:02d}")
    else:
        raise ValueError("You must provide either --year and --month, or --start_year, --start_month, --end_year, --end_month.")

if __name__ == "__main__":
    main()
