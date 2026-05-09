"""
Download hourly weather covariates from Open-Meteo Historical API.

For each country, averages temperature across multiple major cities
to get a more representative national temperature estimate.

Output: data/raw/weather/weather_{country_code}.csv
  - Hourly: temperature, humidity, precipitation, cloud cover, radiation and wind
  - Resampled daily context: temperature, precipitation, radiation, cloud and wind aggregates

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
import argparse

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

HOURLY_WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "sunshine_duration",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_gusts_10m",
]

MAX_RETRIES = 5
BASE_WAIT   = 30  # seconds, doubles each retry


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download hourly Open-Meteo weather covariates")
    p.add_argument("--start_date", default=START_DATE)
    p.add_argument("--end_date", default=END_DATE)
    p.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--force", action="store_true", help="Overwrite existing country CSV files")
    return p.parse_args()


def fetch_city_weather(city_name: str, lat: float, lon: float,
                       start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly weather variables for a single city from Open-Meteo with retry."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_WEATHER_VARIABLES),
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
        frame = pd.DataFrame(index=idx)
        for variable in HOURLY_WEATHER_VARIABLES:
            frame[variable] = pd.Series(hourly[variable], index=idx, dtype=np.float32)
        print(
            f"      → {len(frame)} hours, "
            f"temp={frame['temperature_2m'].min():.1f}°C..{frame['temperature_2m'].max():.1f}°C"
        )
        return frame

    raise RuntimeError(f"Failed to fetch {city_name} after {MAX_RETRIES} retries")


def download_country(country_code: str, info: dict,
                     start_date: str, end_date: str) -> pd.DataFrame:
    """Download and average weather variables across all cities for one country."""
    print(f"\n{'='*60}")
    print(f"  {info['name']} ({country_code}) — {len(info['cities'])} cities")
    print(f"{'='*60}")

    city_frames = []
    for city_name, (lat, lon) in info["cities"].items():
        frame = fetch_city_weather(city_name, lat, lon, start_date, end_date)
        city_frames.append(frame)
        time.sleep(2)  # Respectful pause for the free API

    # Combine and compute national average for each weather variable.
    result = pd.DataFrame(index=city_frames[0].index)
    for variable in HOURLY_WEATHER_VARIABLES:
        result[variable] = pd.concat([frame[variable] for frame in city_frames], axis=1).mean(axis=1)

    # Add daily aggregates (broadcast back to hourly index)
    daily = result["temperature_2m"].resample("D")
    daily_stats = pd.DataFrame({
        "temp_daily_mean": daily.mean(),
        "temp_daily_max":  daily.max(),
        "temp_daily_min":  daily.min(),
    })
    daily_stats["precipitation_daily_sum"] = result["precipitation"].resample("D").sum()
    daily_stats["cloud_cover_daily_mean"] = result["cloud_cover"].resample("D").mean()
    daily_stats["shortwave_radiation_daily_mean"] = result["shortwave_radiation"].resample("D").mean()
    daily_stats["direct_radiation_daily_mean"] = result["direct_radiation"].resample("D").mean()
    daily_stats["diffuse_radiation_daily_mean"] = result["diffuse_radiation"].resample("D").mean()
    daily_stats["sunshine_duration_daily_sum"] = result["sunshine_duration"].resample("D").sum()
    daily_stats["wind_speed_10m_daily_mean"] = result["wind_speed_10m"].resample("D").mean()
    daily_stats["wind_speed_10m_daily_max"] = result["wind_speed_10m"].resample("D").max()
    daily_stats["wind_speed_100m_daily_mean"] = result["wind_speed_100m"].resample("D").mean()
    daily_stats["wind_speed_100m_daily_max"] = result["wind_speed_100m"].resample("D").max()
    daily_stats["wind_gusts_10m_daily_max"] = result["wind_gusts_10m"].resample("D").max()

    result = result.join(daily_stats, how="left")
    daily_cols = list(daily_stats.columns)
    result[daily_cols] = result[daily_cols].ffill()

    print(f"  Result: {len(result)} hours, "
          f"mean={result['temperature_2m'].mean():.1f}°C, "
          f"missing={int(result.isna().sum().sum())}")

    return result


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Open-Meteo Historical Weather Download")
    print(f"Period: {args.start_date} → {args.end_date}")
    print(f"Countries: {len(COUNTRY_CITIES)}")
    print(f"Variables: {', '.join(HOURLY_WEATHER_VARIABLES)}")
    print(f"Output: {args.output_dir}")

    for country_code, info in COUNTRY_CITIES.items():
        out_path = args.output_dir / f"weather_{country_code}.csv"

        # Skip if already downloaded
        if out_path.exists() and not args.force:
            print(f"\n  ✓ {info['name']} ({country_code}) already exists, skipping.")
            continue

        df = download_country(country_code, info, args.start_date, args.end_date)
        df.to_csv(out_path, index_label="utc_timestamp")
        print(f"  Saved → {out_path}")

    print(f"\n✓ All weather data downloaded!")
    print(f"  Total files: {len(COUNTRY_CITIES)}")
    print(f"  Directory: {args.output_dir}")


if __name__ == "__main__":
    main()
