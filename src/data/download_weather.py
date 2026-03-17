"""
Download hourly temperature data from Open-Meteo Historical API.

For each country, averages temperature across multiple major cities
to get a more representative national temperature estimate.

Output: data/raw/weather/weather_{country_code}.csv
  - Hourly: temperature_2m (mean of cities)
  - Resampled daily: temp_mean, temp_max, temp_min

API docs: https://open-meteo.com/en/docs/historical-weather-api
Free, no API key needed for non-commercial use.

Run:
    python src/data/download_weather.py
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────────

# Date range covering train + val + test splits
START_DATE = "2015-01-01"
END_DATE   = "2026-03-05"

# Major cities per country (lat, lon) — chosen for population-weighted coverage
# Multiple cities per country for more representative national temperature
COUNTRY_CITIES = {
    "ES": {
        "name": "Spain",
        "cities": {
            "Madrid":    (40.42, -3.70),
            "Barcelona": (41.39,  2.17),
            "Valencia":  (39.47, -0.38),
            "Sevilla":   (37.39, -5.98),
            "Bilbao":    (43.26, -2.93),
        }
    },
    "FR": {
        "name": "France",
        "cities": {
            "Paris":     (48.86,  2.35),
            "Lyon":      (45.76,  4.84),
            "Marseille": (43.30,  5.37),
            "Toulouse":  (43.60,  1.44),
            "Lille":     (50.63,  3.06),
        }
    },
    "DE": {
        "name": "Germany",
        "cities": {
            "Berlin":    (52.52, 13.41),
            "Munich":    (48.14, 11.58),
            "Hamburg":   (53.55,  9.99),
            "Frankfurt": (50.11, 8.68),
            "Cologne":   (50.94,  6.96),
        }
    },
    "GR": {
        "name": "Greece",
        "cities": {
            "Athens":        (37.98, 23.73),
            "Thessaloniki":  (40.64, 22.94),
            "Patras":        (38.25, 21.73),
        }
    },
    "IT": {
        "name": "Italy",
        "cities": {
            "Rome":    (41.90, 12.50),
            "Milan":   (45.46,  9.19),
            "Naples":  (40.85, 14.27),
            "Turin":   (45.07,  7.69),
            "Palermo": (38.12, 13.36),
        }
    },
    "PT": {
        "name": "Portugal",
        "cities": {
            "Lisbon": (38.72, -9.14),
            "Porto":  (41.15, -8.61),
            "Faro":   (37.02, -7.93),
        }
    },
    "NL": {
        "name": "Netherlands",
        "cities": {
            "Amsterdam": (52.37,  4.90),
            "Rotterdam": (51.92,  4.48),
            "Eindhoven": (51.44,  5.47),
        }
    },
    "BE": {
        "name": "Belgium",
        "cities": {
            "Brussels": (50.85,  4.35),
            "Antwerp":  (51.22,  4.40),
            "Liège":    (50.63,  5.57),
        }
    },
}

API_URL = "https://archive-api.open-meteo.com/v1/archive"

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "raw" / "weather"


MAX_RETRIES = 5
BASE_WAIT   = 30  # seconds, doubles each retry


def fetch_city_temperature(city_name: str, lat: float, lon: float,
                           start_date: str, end_date: str) -> pd.Series:
    """Fetch hourly temperature for a single city from Open-Meteo with retry."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m",
        "timezone": "UTC",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"    Fetching {city_name} ({lat}, {lon})...")
        resp = requests.get(API_URL, params=params, timeout=60)

        if resp.status_code == 429:
            wait = BASE_WAIT * (2 ** (attempt - 1))
            print(f"      ⚠ Rate limited (429). Waiting {wait}s (attempt {attempt}/{MAX_RETRIES})...")
            time.sleep(wait)
            continue

        resp.raise_for_status()
        data = resp.json()

        hourly = data["hourly"]
        idx = pd.to_datetime(hourly["time"], utc=True)
        series = pd.Series(hourly["temperature_2m"], index=idx, name=city_name,
                           dtype=np.float32)
        print(f"      → {len(series)} hours, range: {series.min():.1f}°C to {series.max():.1f}°C")
        return series

    raise RuntimeError(f"Failed to fetch {city_name} after {MAX_RETRIES} retries")


def download_country(country_code: str, info: dict,
                     start_date: str, end_date: str) -> pd.DataFrame:
    """Download and average temperature across all cities for one country."""
    print(f"\n{'='*60}")
    print(f"  {info['name']} ({country_code}) — {len(info['cities'])} cities")
    print(f"{'='*60}")

    city_series = []
    for city_name, (lat, lon) in info["cities"].items():
        s = fetch_city_temperature(city_name, lat, lon, start_date, end_date)
        city_series.append(s)
        time.sleep(2)  # Respectful pause for the free API

    # Combine and compute national average
    cities_df = pd.concat(city_series, axis=1)

    result = pd.DataFrame({
        "temperature_2m": cities_df.mean(axis=1),  # Hourly national mean
    })

    # Add daily aggregates (broadcast back to hourly index)
    daily = result["temperature_2m"].resample("D")
    daily_stats = pd.DataFrame({
        "temp_daily_mean": daily.mean(),
        "temp_daily_max":  daily.max(),
        "temp_daily_min":  daily.min(),
    })
    result = result.join(daily_stats, how="left")
    # Forward-fill the daily stats to hourly granularity
    result[["temp_daily_mean", "temp_daily_max", "temp_daily_min"]] = \
        result[["temp_daily_mean", "temp_daily_max", "temp_daily_min"]].ffill()

    print(f"  Result: {len(result)} hours, "
          f"mean={result['temperature_2m'].mean():.1f}°C, "
          f"missing={result['temperature_2m'].isna().sum()}")

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Open-Meteo Historical Weather Download")
    print(f"Period: {START_DATE} → {END_DATE}")
    print(f"Countries: {len(COUNTRY_CITIES)}")
    print(f"Output: {OUTPUT_DIR}")

    for country_code, info in COUNTRY_CITIES.items():
        out_path = OUTPUT_DIR / f"weather_{country_code}.csv"

        # Skip if already downloaded
        if out_path.exists():
            print(f"\n  ✓ {info['name']} ({country_code}) already exists, skipping.")
            continue

        df = download_country(country_code, info, START_DATE, END_DATE)
        df.to_csv(out_path, index_label="utc_timestamp")
        print(f"  Saved → {out_path}")

    print(f"\n✓ All weather data downloaded!")
    print(f"  Total files: {len(COUNTRY_CITIES)}")
    print(f"  Directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
