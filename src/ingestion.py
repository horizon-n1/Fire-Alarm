import os
import requests
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import time

load_dotenv()

API_KEY = os.getenv("NASA_FIRMS_API_KEY")
BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# NRT sources are capped at 5 days; standard sources allow up to 10
MAX_DAYS_NRT = 5
MAX_DAYS_STANDARD = 10

NRT_SOURCES = {"VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "MODIS_NRT"}

def fetch_active_fires(region_box, days=1, source="VIIRS_SNPP_NRT"):
    """
    Fetches NASA FIRMS active fire data for a specific bounding box.
    Automatically clamps days to the API limit for the given source.
    """
    if not API_KEY:
        raise ValueError("NASA FIRMS API Key not found in .env file.")

    max_days = MAX_DAYS_NRT if source in NRT_SOURCES else MAX_DAYS_STANDARD

    if days > max_days:
        print(f"Warning: Requested {days} days exceeds limit of {max_days} for '{source}'. Clamping to {max_days}.")
        days = max_days

    return _fetch_single(region_box, days, source)


def _fetch_single(region_box, days, source):
    """Makes a single API call for a valid day range."""
    print(f"Requesting {days} day(s) of {source} fire data...")
    endpoint = f"{BASE_URL}/{API_KEY}/{source}/{region_box}/{days}"

    try:
        response = requests.get(endpoint)
        response.raise_for_status()

        if not response.text.strip() or "No fire data" in response.text:
            print("No active fires found in this region/time window.")
            return pd.DataFrame()

        lines = response.text.strip().split('\n')
        headers = lines[0].split(',')
        data = [line.split(',') for line in lines[1:] if line.strip()]
        df = pd.DataFrame(data, columns=headers)

        df['latitude'] = df['latitude'].astype(float)
        df['longitude'] = df['longitude'].astype(float)
        df['bright_ti4'] = pd.to_numeric(df['bright_ti4'], errors='coerce')

        print(f"  Retrieved {len(df)} fire data points.")
        return df

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error {response.status_code}: {e}")
        print(f"Response body: {response.text[:300]}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"API Request Failed: {e}")
        return None


def _fetch_chunked(region_box, total_days, source, chunk_size):
    """
    Splits a large day range into chunk_size chunks and concatenates results.
    Note: FIRMS 'days' always counts back from TODAY, so chunks will overlap.
    Use this for broad data gathering, not precise date windowing.
    """
    frames = []
    for chunk_days in range(chunk_size, total_days + 1, chunk_size):
        chunk_days = min(chunk_days, total_days)
        df = _fetch_single(region_box, chunk_days, source)
        if df is not None and not df.empty:
            frames.append(df)
        time.sleep(1)  # Be polite to the API

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames).drop_duplicates()
    print(f"Total unique records after chunking: {len(combined)}")
    return combined


def save_data(df, filename="active_fires.csv"):
    """Saves DataFrame to the data/raw/ directory."""
    if df is None or df.empty:
        print("No data to save.")
        return

    output_dir = Path(__file__).parent.parent / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / filename
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")


if __name__ == "__main__":
    california_bbox = "-124.5,32.5,-114.0,42.0"

    print("Starting Data Ingestion Pipeline...")
    fire_data = fetch_active_fires(region_box=california_bbox, days=10)
    save_data(fire_data)
    print("Pipeline Complete.")