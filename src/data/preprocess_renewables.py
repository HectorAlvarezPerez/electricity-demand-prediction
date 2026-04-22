"""Build the daily renewables forecasting dataset.

The output is intentionally separate from the demand dataset:
`data/processed_renewables_daily/{train,val,test}.parquet`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.renewables import (
    DAILY_EXTERNAL_COLUMNS,
    RENEWABLE_TARGET_COLS,
    add_calendar_features,
    add_country_dummies,
    add_day_ahead_targets,
    add_lag_features,
    feature_columns,
    load_all_generation_daily,
    split_by_target_date,
    target_columns,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Preprocess daily ENTSO-E renewables data")
    p.add_argument("--generation_dir", type=Path, default=root / "data" / "raw" / "europe" / "generation")
    p.add_argument("--weather_dir", type=Path, default=root / "data" / "raw" / "weather_enriched")
    p.add_argument("--output_dir", type=Path, default=root / "data" / "processed_renewables_daily")
    p.add_argument("--include_external", action="store_true")
    return p.parse_args()


def load_enriched_weather(weather_dir: Path) -> pd.DataFrame:
    parts = []
    for path in sorted(weather_dir.glob("weather_daily_*.csv")):
        code = path.stem.split("_")[-1]
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df["country_code"] = code
        missing = [col for col in DAILY_EXTERNAL_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing weather columns: {missing}")
        parts.append(df[["date", "country_code", *DAILY_EXTERNAL_COLUMNS]])
    if not parts:
        raise FileNotFoundError(f"No weather_daily_*.csv files found in {weather_dir}")
    return pd.concat(parts, ignore_index=True)


def attach_target_day_weather(df: pd.DataFrame, weather_dir: Path) -> pd.DataFrame:
    weather = load_enriched_weather(weather_dir)
    out = df.copy()
    out["weather_date"] = pd.to_datetime(out["date"]) + pd.Timedelta(days=1)
    out = out.merge(
        weather.rename(columns={"date": "weather_date"}),
        on=["country_code", "weather_date"],
        how="left",
    )
    return out.drop(columns=["weather_date"])


def build_dataset(
    generation_dir: Path,
    *,
    weather_dir: Path | None = None,
    include_external: bool = False,
) -> pd.DataFrame:
    df = load_all_generation_daily(generation_dir)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_day_ahead_targets(df)
    if include_external:
        if weather_dir is None:
            raise ValueError("weather_dir is required when include_external=True")
        df = attach_target_day_weather(df, weather_dir)
    df = add_country_dummies(df)

    required = [
        *feature_columns(
            df,
            include_temporal=True,
            include_external=include_external,
            include_country_id=True,
        ),
        *target_columns(),
        "target_date",
    ]
    df = df.dropna(subset=required).reset_index(drop=True)
    return df.sort_values(["country_code", "date"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = build_dataset(
        args.generation_dir,
        weather_dir=args.weather_dir if args.include_external else None,
        include_external=args.include_external,
    )
    print(f"Built renewables dataset: shape={df.shape}")
    print(f"Date range: {df['date'].min()} -> {df['date'].max()}")
    print(f"Target date range: {df['target_date'].min()} -> {df['target_date'].max()}")
    print(f"Countries: {sorted(df['country_code'].unique().tolist())}")

    splits = split_by_target_date(df)
    for name, split_df in splits.items():
        path = args.output_dir / f"{name}.parquet"
        split_df.to_parquet(path, index=False)
        print(f"Saved {name}: {len(split_df):,} rows -> {path}")


if __name__ == "__main__":
    main()
