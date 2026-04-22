"""Download enriched daily weather features from Open-Meteo.

The country aggregation follows the existing demand weather downloader:
download several representative cities per country and average them into
a national proxy.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.data.download_weather import COUNTRY_CITIES
from src.data.renewables import DAILY_EXTERNAL_COLUMNS


API_URL = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "2015-01-01"
END_DATE = "2026-03-05"
MAX_RETRIES = 5
BASE_WAIT_SECONDS = 30

HOURLY_VARIABLES = [
    "cloud_cover",
    "wind_speed_100m",
    "wind_gusts_10m",
]

DAILY_VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "shortwave_radiation_sum",
    "precipitation_sum",
    "daylight_duration",
    "sunshine_duration",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Download enriched daily Open-Meteo weather")
    p.add_argument("--start_date", default=START_DATE)
    p.add_argument("--end_date", default=END_DATE)
    p.add_argument("--output_dir", type=Path, default=root / "data" / "raw" / "weather_enriched")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def fetch_city_weather(city_name: str, lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
        "hourly": ",".join(HOURLY_VARIABLES),
        "daily": ",".join(DAILY_VARIABLES),
    }
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"    Fetching {city_name} ({lat}, {lon})...")
        resp = requests.get(API_URL, params=params, timeout=60)
        if resp.status_code == 429:
            wait = BASE_WAIT_SECONDS * (2 ** (attempt - 1))
            print(f"      Rate limited. Waiting {wait}s ({attempt}/{MAX_RETRIES})")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        payload = resp.json()
        daily = pd.DataFrame(payload["daily"])
        daily["date"] = pd.to_datetime(daily["time"])
        daily = daily.drop(columns=["time"])

        hourly = pd.DataFrame(payload["hourly"])
        hourly["date"] = pd.to_datetime(hourly["time"], utc=True).dt.floor("D").dt.tz_localize(None)
        hourly_daily = hourly.groupby("date", as_index=False).agg(
            cloud_cover_mean=("cloud_cover", "mean"),
            wind_speed_100m_mean=("wind_speed_100m", "mean"),
            wind_speed_100m_max=("wind_speed_100m", "max"),
            wind_gusts_10m_max=("wind_gusts_10m", "max"),
        )
        out = daily.merge(hourly_daily, on="date", how="left")
        for col in DAILY_EXTERNAL_COLUMNS:
            if col not in out.columns:
                out[col] = np.nan
        return out[["date", *DAILY_EXTERNAL_COLUMNS]]
    raise RuntimeError(f"Failed to fetch {city_name} after {MAX_RETRIES} retries")


def download_country(country_code: str, info: dict, start_date: str, end_date: str) -> pd.DataFrame:
    print(f"\n{info['name']} ({country_code}) - {len(info['cities'])} cities")
    city_frames = []
    for city_name, (lat, lon) in info["cities"].items():
        city_frames.append(fetch_city_weather(city_name, lat, lon, start_date, end_date))
        time.sleep(2)

    stacked = pd.concat(
        [frame.assign(city=f"city_{idx}") for idx, frame in enumerate(city_frames)],
        ignore_index=True,
    )
    numeric_cols = [c for c in DAILY_EXTERNAL_COLUMNS if c in stacked.columns]
    country = stacked.groupby("date", as_index=False)[numeric_cols].mean()
    missing = country[numeric_cols].isna().sum()
    if int(missing.sum()) > 0:
        raise ValueError(f"Missing weather values for {country_code}: {missing[missing > 0].to_dict()}")
    return country


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Open-Meteo enriched daily weather: {args.start_date} -> {args.end_date}")
    for country_code, info in COUNTRY_CITIES.items():
        out_path = args.output_dir / f"weather_daily_{country_code}.csv"
        if out_path.exists() and not args.overwrite:
            print(f"Skipping {country_code}: {out_path} already exists")
            continue
        df = download_country(country_code, info, args.start_date, args.end_date)
        df.to_csv(out_path, index=False)
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
