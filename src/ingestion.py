import os
import requests
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

API_KEY  = os.getenv("NASA_FIRMS_API_KEY")
BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

NRT_SOURCES      = {"VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "MODIS_NRT"}
MAX_DAYS_NRT     = 5
MAX_DAYS_ARCHIVE = 10   # archive endpoint still caps per-call at 10 days


# ════════════════════════════════════════════════════════════════════════════
# Core fetch — single call (≤10 days or a specific start_date)
# ════════════════════════════════════════════════════════════════════════════

def fetch_active_fires(
    region_box: str,
    days:       int  = 10,
    source:     str  = "VIIRS_SNPP_SP",
    start_date: str  = None          # "YYYY-MM-DD" — triggers archive mode
) -> pd.DataFrame:
    """
    Fetches NASA FIRMS fire data.

    Two modes:
        NRT mode    — omit start_date, uses rolling 'days' window from today
        Archive mode — provide start_date ("YYYY-MM-DD"), fetches 'days' days
                       forward from that date (max 10 per call)
    """
    if not API_KEY:
        raise ValueError("NASA_FIRMS_API_KEY not found in .env")

    if source in NRT_SOURCES and days > MAX_DAYS_NRT:
        print(f"Warning: Clamping {days} days to {MAX_DAYS_NRT} for NRT source '{source}'.")
        days = MAX_DAYS_NRT

    if start_date:
        # Archive endpoint format:
        # /api/area/csv/{key}/{source}/{bbox}/{days}/{start_date}
        endpoint = f"{BASE_URL}/{API_KEY}/{source}/{region_box}/{days}/{start_date}"
        print(f"Archive fetch | source={source} | start={start_date} | days={days}")
    else:
        endpoint = f"{BASE_URL}/{API_KEY}/{source}/{region_box}/{days}"
        print(f"NRT fetch     | source={source} | days={days}")

    try:
        response = requests.get(endpoint, timeout=30)
        response.raise_for_status()

        text = response.text.strip()
        if not text or "No fire data" in text:
            print("  → No fire data returned for this window.")
            return pd.DataFrame()

        lines   = text.split('\n')
        headers = lines[0].split(',')
        data    = [line.split(',') for line in lines[1:] if line.strip()]
        df      = pd.DataFrame(data, columns=headers)

        float_cols = ["latitude", "longitude", "bright_ti4", "frp"]
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        print(f"  → {len(df)} detections retrieved.")
        return df

    except requests.exceptions.HTTPError as e:
        print(f"HTTP {response.status_code}: {e} | {response.text[:200]}")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# Multi-window fetch — stitches multiple 10-day archive calls together
# ════════════════════════════════════════════════════════════════════════════

def fetch_fire_season(
    region_box:  str,
    season_start: str,   # "YYYY-MM-DD"
    season_end:   str,   # "YYYY-MM-DD"
    sources:     list = ["VIIRS_SNPP_SP", "MODIS_SP"]
) -> pd.DataFrame:
    """
    Fetches an entire fire season by chunking into 10-day archive windows
    across multiple sources and deduplicating.

    Example:
        fetch_fire_season(bbox, "2020-08-01", "2020-09-30")
        → 4 chunks × 2 sources = 8 API calls covering 60 days
    """
    from datetime import datetime, timedelta
    import time

    start = datetime.strptime(season_start, "%Y-%m-%d")
    end   = datetime.strptime(season_end,   "%Y-%m-%d")

    frames = []
    chunk_start = start

    while chunk_start < end:
        chunk_end  = min(chunk_start + timedelta(days=10), end)
        days_in_chunk = (chunk_end - chunk_start).days
        date_str   = chunk_start.strftime("%Y-%m-%d")

        for source in sources:
            df = fetch_active_fires(
                region_box = region_box,
                days       = days_in_chunk,
                source     = source,
                start_date = date_str
            )
            if not df.empty:
                df["source"] = source          # track which sensor saw it
                frames.append(df)

        time.sleep(1)                          # be polite to the API
        chunk_start += timedelta(days=10)

    if not frames:
        print("No data retrieved for the entire season window.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Deduplicate on position — VIIRS and MODIS overlap spatially
    before = len(combined)
    combined = combined.drop_duplicates(subset=["latitude", "longitude"]).reset_index(drop=True)
    print(f"\nSeason fetch complete: {before} total → {len(combined)} unique detections.")
    return combined


# ════════════════════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════════════════════

def save_data(df: pd.DataFrame, filename: str = "active_fires.csv"):
    if df is None or df.empty:
        print("No data to save.")
        return

    output_dir = Path(__file__).parent.parent / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / filename
    df.to_csv(file_path, index=False)
    print(f"Saved → {file_path}  ({len(df)} rows)")


# ════════════════════════════════════════════════════════════════════════════
# Entry point — 2020 California Fire Season (historically worst on record)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 2020 California fire season — worst in recorded state history
    # August–October 2020 saw the SCU, LNU, Creek, and August Complex fires
    # which collectively burned over 4 million acres
    CALIFORNIA_BBOX = "-124.5,32.5,-114.0,42.0"

    print("═══ PyroFlow Historical Ingestion ═══")
    print("Target: 2020 California Fire Season (Aug–Oct)")
    print("Sources: VIIRS_SNPP_SP + MODIS_SP\n")

    fire_df = fetch_fire_season(
        region_box   = CALIFORNIA_BBOX,
        season_start = "2020-08-01",
        season_end   = "2020-10-31",
        sources      = ["VIIRS_SNPP_SP", "MODIS_SP"]
    )

    save_data(fire_df, filename="active_fires.csv")
    print("\n═══ Ingestion Complete ═══")