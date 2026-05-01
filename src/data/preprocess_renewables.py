"""Build the hourly renewables forecasting dataset.

The output is intentionally separate from the demand dataset:
`data/processed_renewables_hourly/{train,val,test}.parquet`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.renewables import (
    HOURLY_EXTERNAL_COLUMNS,
    add_calendar_features,
    add_country_dummies,
    add_hour_ahead_targets,
    add_lag_features,
    feature_columns,
    load_all_generation_hourly,
    split_by_target_timestamp,
    target_columns,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Preprocess hourly ENTSO-E renewables data")
    p.add_argument("--generation_dir", type=Path, default=root / "data" / "raw" / "europe" / "generation")
    p.add_argument("--weather_dir", type=Path, default=root / "data" / "raw" / "weather")
    p.add_argument("--output_dir", type=Path, default=root / "data" / "processed_renewables_hourly")
    p.add_argument("--include_external", action="store_true")
    return p.parse_args()


def load_hourly_weather(weather_dir: Path) -> pd.DataFrame:
    parts = []
    for path in sorted(weather_dir.glob("weather_*.csv")):
        code = path.stem.split("_")[-1]
        df = pd.read_csv(path)
        df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True)
        df["country_code"] = code
        missing = [col for col in HOURLY_EXTERNAL_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing weather columns: {missing}")
        parts.append(df[["utc_timestamp", "country_code", *HOURLY_EXTERNAL_COLUMNS]])
    if not parts:
        raise FileNotFoundError(f"No weather_*.csv files found in {weather_dir}")
    return pd.concat(parts, ignore_index=True)


def attach_target_hour_weather(df: pd.DataFrame, weather_dir: Path) -> pd.DataFrame:
    weather = load_hourly_weather(weather_dir)
    out = df.copy()
    out["weather_timestamp"] = pd.to_datetime(out["utc_timestamp"], utc=True) + pd.Timedelta(hours=1)
    out = out.merge(
        weather.rename(columns={"utc_timestamp": "weather_timestamp"}),
        on=["country_code", "weather_timestamp"],
        how="left",
    )
    return out.drop(columns=["weather_timestamp"])


# Backward-compatible alias while the repo moves from D+1 to H+1.
attach_target_day_weather = attach_target_hour_weather


def build_dataset(
    generation_dir: Path,
    *,
    weather_dir: Path | None = None,
    include_external: bool = False,
) -> pd.DataFrame:
    df = load_all_generation_hourly(generation_dir)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_hour_ahead_targets(df)
    if include_external:
        if weather_dir is None:
            raise ValueError("weather_dir is required when include_external=True")
        df = attach_target_hour_weather(df, weather_dir)
    df = add_country_dummies(df)

    required = [
        *feature_columns(
            df,
            include_temporal=True,
            include_external=include_external,
            include_country_id=True,
        ),
        *target_columns(),
        "target_timestamp",
    ]
    df = df.dropna(subset=required).reset_index(drop=True)
    return df.sort_values(["country_code", "utc_timestamp"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = build_dataset(
        args.generation_dir,
        weather_dir=args.weather_dir if args.include_external else None,
        include_external=args.include_external,
    )
    print(f"Built renewables dataset: shape={df.shape}")
    print(f"Timestamp range: {df['utc_timestamp'].min()} -> {df['utc_timestamp'].max()}")
    print(f"Target timestamp range: {df['target_timestamp'].min()} -> {df['target_timestamp'].max()}")
    print(f"Countries: {sorted(df['country_code'].unique().tolist())}")

    splits = split_by_target_timestamp(df)
    for name, split_df in splits.items():
        path = args.output_dir / f"{name}.parquet"
        split_df.to_parquet(path, index=False)
        print(f"Saved {name}: {len(split_df):,} rows -> {path}")


if __name__ == "__main__":
    main()
