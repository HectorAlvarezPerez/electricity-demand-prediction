"""
Download ENTSO-E day-ahead load forecasts for the project countries.

Output:
    data/raw/europe/forecast/entsoe_load_forecast_{country}.csv

The script stores delivery-time forecasts at hourly resolution in UTC so they
can be compared against the project's day-ahead (h=24) predictions.
"""
from __future__ import annotations

import os
import time
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient

COUNTRIES = ["ES", "FR", "DE", "GR", "IT", "PT", "NL", "BE"]
DEFAULT_START_YEAR = 2024
SLEEP_SECONDS = 1


def _get_client() -> EntsoePandasClient:
    api_key = os.environ.get("ENTSOE_TOKEN_KEY")
    if not api_key:
        raise RuntimeError(
            "ENTSOE_TOKEN_KEY is not set. Export the token before running this script."
        )
    return EntsoePandasClient(api_key=api_key)


def _to_forecast_frame(obj: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(obj, pd.Series):
        df = obj.to_frame(name="forecast")
    else:
        df = obj.copy()
        if len(df.columns) == 1:
            df.columns = ["forecast"]
        else:
            first = df.columns[0]
            df = df[[first]].rename(columns={first: "forecast"})

    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.resample("1h").mean()
    df.index.name = "utc_timestamp"
    return df


def download_country_forecast(
    client: EntsoePandasClient,
    country_code: str,
    output_dir: Path,
    *,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int | None = None,
) -> Path:
    if end_year is None:
        end_year = datetime.now().year

    parts: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        if year < end_year:
            end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
        else:
            end = pd.Timestamp(datetime.now(), tz="UTC")

        print(f"[{country_code}] downloading day-ahead load forecast for {year}...", flush=True)
        try:
            data = client.query_load_forecast(country_code, start=start, end=end)
            df = _to_forecast_frame(data)
            if not df.empty:
                parts.append(df)
        except Exception as exc:  # pragma: no cover - depends on network/API
            print(f"[{country_code}] {year} failed: {exc}", flush=True)
        time.sleep(SLEEP_SECONDS)

    if not parts:
        raise RuntimeError(f"No forecast data downloaded for {country_code}.")

    final = pd.concat(parts).sort_index()
    final = final[~final.index.duplicated(keep="last")]
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"entsoe_load_forecast_{country_code}.csv"
    final.to_csv(out_path)
    print(f"[{country_code}] saved -> {out_path}", flush=True)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download ENTSO-E day-ahead load forecasts")
    p.add_argument("--start_year", type=int, default=DEFAULT_START_YEAR)
    p.add_argument("--end_year", type=int, default=datetime.now().year)
    p.add_argument("--countries", nargs="+", default=COUNTRIES)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    output_dir = root / "data" / "raw" / "europe" / "forecast"
    client = _get_client()

    for code in args.countries:
        download_country_forecast(
            client,
            code,
            output_dir,
            start_year=args.start_year,
            end_year=args.end_year,
        )

    print("\nAll ENTSO-E day-ahead load forecasts downloaded.", flush=True)


if __name__ == "__main__":
    main()
