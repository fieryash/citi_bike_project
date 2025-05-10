import pandas as pd
from pathlib import Path
import re

# List all cleaned parquet files
cleaned_dir = Path("tmp_raw/cleaned")
files = list(cleaned_dir.glob("*_cleaned.parquet"))

# Extract years from filenames
years = set()
year_pattern = re.compile(r"(20\d{2})")
for f in files:
    match = year_pattern.search(f.name)
    if match:
        years.add(int(match.group(1)))

if years:
    print(f"Years covered in cleaned features: {sorted(years)}")
else:
    print("No years found in cleaned features filenames.")
