"""
Preprocessing pipeline for the long-format forecasting dataset.

The new representation keeps one row per (utc_timestamp, country_code) and
generates explicit autoregressive lag columns per country instead of
materialising full sliding windows and flattening them later.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:  # pragma: no cover - handled at runtime with a clear error
    pl = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_CODE = "ES"
SOURCE_CODES = ["BE", "DE", "FR", "GR", "IT", "NL", "PT"]
ALL_CODES = [TARGET_CODE, *SOURCE_CODES]

COUNTRY_TIMEZONES = {
    "ES": "Europe/Madrid",
    "BE": "Europe/Brussels",
    "DE": "Europe/Berlin",
    "FR": "Europe/Paris",
    "GR": "Europe/Athens",
    "IT": "Europe/Rome",
    "NL": "Europe/Amsterdam",
    "PT": "Europe/Lisbon",
}

WEATHER_FEATURES = [
    "temperature_2m",
    "temp_daily_mean",
    "temp_daily_max",
    "temp_daily_min",
]

TRAIN_START = pd.Timestamp("2015-01-01", tz="UTC")
TRAIN_END = pd.Timestamp("2022-12-31 23:00:00", tz="UTC")
VAL_START = pd.Timestamp("2023-01-01", tz="UTC")
VAL_END = pd.Timestamp("2023-12-31 23:00:00", tz="UTC")
TEST_START = pd.Timestamp("2024-01-01", tz="UTC")

DEMAND_LAGS = list(range(1, 25)) + [48, 168]
DEFAULT_PRED_LEN = 24

META_COLUMNS = ["utc_timestamp", "country_code", "role"]
TEMPORAL_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
]


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------
def _require_polars() -> None:
    if pl is None:  # pragma: no cover - depends on user env
        raise ImportError(
            "polars is required for the long-format preprocessing pipeline. "
            "Install dependencies with 'pip install -r requirements.txt'."
        )


def role_for_code(code: str) -> str:
    return "target" if code.upper() == TARGET_CODE else "source"


def target_columns(pred_len: int = DEFAULT_PRED_LEN) -> list[str]:
    return [f"y_h{i}" for i in range(1, pred_len + 1)]


def lag_columns() -> list[str]:
    return [f"lag_{lag}" for lag in DEMAND_LAGS]


def country_id_columns(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.startswith("country_id_")])


def weather_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in WEATHER_FEATURES if c in df.columns]


def feature_columns(
    df: pd.DataFrame,
    *,
    include_temporal: bool = False,
    include_weather: bool = False,
    include_country_id: bool = False,
) -> list[str]:
    cols = ["demand", *[c for c in lag_columns() if c in df.columns]]
    if include_temporal:
        cols.extend([c for c in TEMPORAL_COLUMNS if c in df.columns])
    if include_weather:
        cols.extend(weather_columns(df))
    if include_country_id:
        cols.extend(country_id_columns(df))
    return cols


def normalize_data(
    df: pd.DataFrame,
    method: str = "standard",
    params: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict]:
    """Column-wise normalisation for numeric pandas dataframes."""
    df = df.copy()

    if method == "standard":
        if params is None:
            params = {
                "method": "standard",
                "mean": df.mean(),
                "std": df.std().replace(0, 1),
            }
        df = (df - params["mean"]) / params["std"]

    elif method == "minmax":
        if params is None:
            params = {
                "method": "minmax",
                "min": df.min(),
                "max": df.max(),
            }
        denom = (params["max"] - params["min"]).replace(0, 1)
        df = (df - params["min"]) / denom

    else:
        raise ValueError(f"Unknown normalisation method: {method!r}")

    return df, params


def denormalize_data(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    if params["method"] == "standard":
        df = df * params["std"] + params["mean"]
    elif params["method"] == "minmax":
        denom = params["max"] - params["min"]
        df = df * denom + params["min"]
    return df


# ---------------------------------------------------------------------------
# Raw loading and per-country alignment
# ---------------------------------------------------------------------------
def _load_country_demand(demand_dir: Path, code: str) -> pd.DataFrame:
    path = demand_dir / f"entsoe_demand_{code}.csv"
    df = pd.read_csv(path)
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True)
    df = df.rename(columns={"demand": "demand"})
    return (
        df[["utc_timestamp", "demand"]]
        .drop_duplicates(subset=["utc_timestamp"])
        .set_index("utc_timestamp")
        .sort_index()
    )


def _load_country_weather(weather_dir: Path, code: str) -> pd.DataFrame:
    path = weather_dir / f"weather_{code}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True)
    cols = ["utc_timestamp", *WEATHER_FEATURES]
    return df[cols].set_index("utc_timestamp").sort_index()


def add_local_temporal_features(frame: pd.DataFrame, code: str) -> pd.DataFrame:
    """Add local-time temporal features for a single-country dataframe."""
    frame = frame.copy()
    idx_local = frame.index.tz_convert(COUNTRY_TIMEZONES[code])

    hour = idx_local.hour
    dow = idx_local.dayofweek
    month = idx_local.month - 1

    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    frame["dow_sin"] = np.sin(2 * np.pi * dow / 7).astype(np.float32)
    frame["dow_cos"] = np.cos(2 * np.pi * dow / 7).astype(np.float32)
    frame["month_sin"] = np.sin(2 * np.pi * month / 12).astype(np.float32)
    frame["month_cos"] = np.cos(2 * np.pi * month / 12).astype(np.float32)
    frame["is_weekend"] = (idx_local.dayofweek >= 5).astype(np.int8)
    return frame


def build_long_base_df(
    demand_dir: Path,
    weather_dir: Optional[Path] = None,
    *,
    include_temporal: bool = True,
) -> pd.DataFrame:
    """
    Build the canonical long-format dataframe before explicit lags/horizons.

    Each row corresponds to one country and one UTC hour. Missing demand/weather
    values are interpolated up to 3 consecutive hours within each country.
    """
    demand_frames = {code: _load_country_demand(demand_dir, code) for code in ALL_CODES}
    weather_frames = (
        {code: _load_country_weather(weather_dir, code) for code in ALL_CODES}
        if weather_dir is not None
        else {}
    )

    overlap_start = max(frame.index.min() for frame in demand_frames.values())
    overlap_end = min(frame.index.max() for frame in demand_frames.values())
    hourly_index = pd.date_range(overlap_start, overlap_end, freq="h", tz="UTC")

    parts: list[pd.DataFrame] = []
    for code in ALL_CODES:
        frame = demand_frames[code].loc[overlap_start:overlap_end].copy()
        frame = frame.reindex(hourly_index)
        frame = frame.interpolate(method="time", limit=3, limit_area="inside")

        if weather_frames:
            weather = weather_frames[code].loc[overlap_start:overlap_end].copy()
            weather = weather.reindex(hourly_index)
            weather = weather.interpolate(method="time", limit=3, limit_area="inside")
            frame = frame.join(weather)

        frame = frame.dropna().copy()
        frame["country_code"] = code
        frame["role"] = role_for_code(code)
        if include_temporal:
            frame = add_local_temporal_features(frame, code)

        frame.index.name = "utc_timestamp"
        parts.append(frame.reset_index())

    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["country_code", "utc_timestamp"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Long-format lag generation with Polars
# ---------------------------------------------------------------------------
def _downcast_polars_numeric(df: "pl.DataFrame") -> "pl.DataFrame":
    float_cols = [
        c for c, dtype in df.schema.items()
        if dtype in (pl.Float64, pl.Float32)
    ]
    int_cols = [
        c for c, dtype in df.schema.items()
        if dtype in (pl.Int64, pl.Int32, pl.Int16)
    ]

    exprs = [pl.col(c).cast(pl.Float32) for c in float_cols]
    exprs.extend(
        pl.col(c).cast(pl.Int8)
        for c in int_cols
        if c == "is_weekend" or c.startswith("country_id_")
    )
    if exprs:
        df = df.with_columns(exprs)
    return df


def build_long_model_df(
    demand_dir: Path,
    weather_dir: Optional[Path] = None,
    *,
    pred_len: int = DEFAULT_PRED_LEN,
    include_temporal: bool = True,
) -> pd.DataFrame:
    _require_polars()

    base_df = build_long_base_df(
        demand_dir,
        weather_dir=weather_dir,
        include_temporal=include_temporal,
    )
    base_pl = pl.from_pandas(base_df)

    lag_exprs = [
        pl.col("demand").shift(lag).over("country_code").alias(f"lag_{lag}")
        for lag in DEMAND_LAGS
    ]
    target_exprs = [
        pl.col("demand").shift(-h).over("country_code").alias(f"y_h{h}")
        for h in range(1, pred_len + 1)
    ]

    model_df = (
        base_pl.lazy()
        .sort(["country_code", "utc_timestamp"])
        .with_columns(lag_exprs + target_exprs)
        .collect()
    )
    model_df = model_df.drop_nulls(subset=[*lag_columns(), *target_columns(pred_len)])
    model_df = _downcast_polars_numeric(model_df)

    pdf = model_df.to_pandas()
    pdf["utc_timestamp"] = pd.to_datetime(pdf["utc_timestamp"], utc=True)
    country_dummies = pd.get_dummies(pdf["country_code"].str.lower(), prefix="country_id").astype(np.int8)
    pdf = pd.concat([pdf, country_dummies], axis=1)
    pdf = pdf.sort_values(["country_code", "utc_timestamp"]).reset_index(drop=True)
    return pdf


def split_by_time(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the long-format dataframe by timestamp."""
    ts = pd.to_datetime(df["utc_timestamp"], utc=True)
    return {
        "train": df.loc[(ts >= TRAIN_START) & (ts <= TRAIN_END)].copy(),
        "val": df.loc[(ts >= VAL_START) & (ts <= VAL_END)].copy(),
        "test": df.loc[ts >= TEST_START].copy(),
    }


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    _require_polars()

    root = Path(__file__).resolve().parents[2]
    demand_dir = root / "data" / "raw" / "europe" / "demand"
    weather_dir = root / "data" / "raw" / "weather"
    out_dir = root / "data" / "processed_long"
    out_dir.mkdir(parents=True, exist_ok=True)

    use_weather = weather_dir.exists()
    print("Building long-format model dataframe...")
    df = build_long_model_df(
        demand_dir,
        weather_dir=weather_dir if use_weather else None,
        pred_len=DEFAULT_PRED_LEN,
        include_temporal=True,
    )
    print(f"  Shape: {df.shape}")
    print(f"  Range: {df['utc_timestamp'].min()} -> {df['utc_timestamp'].max()}")
    print(f"  Countries: {sorted(df['country_code'].unique().tolist())}")

    splits = split_by_time(df)
    for name, split_df in splits.items():
        path = out_dir / f"{name}.parquet"
        split_df.to_parquet(path, index=False)
        print(f"  Saved {name}: {len(split_df):,} rows -> {path}")

    print("Done.")


if __name__ == "__main__":
    main()
