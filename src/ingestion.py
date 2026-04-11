import os
import requests
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

API_KEY = os.getenv("NASA_FIRMS_API_KEY")
BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

def fetch_active_fires(region_box, days=1, source="VIIRS_SNPP_NRT"):
    """Fetches NASA FIRMS active fire data for a specific bounding box."""
    if not API_KEY:
        raise ValueError("NASA FIRMS API Key not found in .env file.")

    print(f"Requesting {days} day(s) of {source} fire data...")
    endpoint = f"{BASE_URL}/{API_KEY}/{source}/{region_box}/{days}"
    
    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        
        if response.text.strip() == "No fire data found":
            print("No active fires found in this region.")
            return pd.DataFrame()

        # Parse CSV text into a DataFrame
        lines = response.text.strip().split('\n')
        headers = lines[0].split(',')
        data = [line.split(',') for line in lines[1:]]
        df = pd.DataFrame(data, columns=headers)
        
        # Typecast coordinates and brightness for processing
        df['latitude'] = df['latitude'].astype(float)
        df['longitude'] = df['longitude'].astype(float)
        df['bright_ti4'] = df['bright_ti4'].astype(float)
        
        print(f"Successfully retrieved {len(df)} fire data points.")
        return df

    except requests.exceptions.RequestException as e:
        print(f"API Request Failed: {e}")
        return None

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
    # Bounding box for California (West, South, East, North)
    california_bbox = "-124.5,32.5,-114.0,42.0"
    
    print("Starting Data Ingestion Pipeline...")
    fire_data = fetch_active_fires(region_box=california_bbox, days=1)
    save_data(fire_data)
    print("Pipeline Complete.")