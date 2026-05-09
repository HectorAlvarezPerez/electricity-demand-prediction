"""Shared helpers for the hourly renewables forecasting dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import (
    ALL_CODES,
    COUNTRY_TIMEZONES,
    TARGET_CODE,
    TRAIN_END,
    TRAIN_START,
    TEST_START,
    VAL_END,
    VAL_START,
    role_for_code,
)


RENEWABLE_TARGET_COLS = [
    "solar_mwh",
    "wind_mwh",
    "hydro_mwh",
    "renewable_total_mwh",
]

RENEWABLE_COMPONENT_COLS = [
    "solar_mwh",
    "wind_onshore_mwh",
    "wind_offshore_mwh",
    "hydro_run_of_river_mwh",
    "hydro_reservoir_mwh",
    "biomass_mwh",
    "geothermal_mwh",
    "marine_mwh",
    "other_renewable_mwh",
]

RENEWABLE_TECH_PREFIXES = {
    "solar_mwh": ("Solar",),
    "wind_onshore_mwh": ("Wind Onshore",),
    "wind_offshore_mwh": ("Wind Offshore",),
    "hydro_run_of_river_mwh": ("Hydro Run-of-river and poundage",),
    "hydro_reservoir_mwh": ("Hydro Water Reservoir",),
    "biomass_mwh": ("Biomass",),
    "geothermal_mwh": ("Geothermal",),
    "marine_mwh": ("Marine",),
    "other_renewable_mwh": ("Other renewable",),
}

DEFAULT_RENEWABLE_LAGS = [1, 2, 24, 48, 168]
DEFAULT_RENEWABLE_ROLL_WINDOWS = [24, 168]

DAILY_EXTERNAL_COLUMNS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "shortwave_radiation_sum",
    "daylight_duration",
    "sunshine_duration",
    "cloud_cover_mean",
    "precipitation_sum",
    "wind_speed_100m_mean",
    "wind_speed_100m_max",
    "wind_gusts_10m_max",
]

HOURLY_EXTERNAL_COLUMNS = [
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
    "temp_daily_mean",
    "temp_daily_max",
    "temp_daily_min",
    "precipitation_daily_sum",
    "cloud_cover_daily_mean",
    "shortwave_radiation_daily_mean",
    "direct_radiation_daily_mean",
    "diffuse_radiation_daily_mean",
    "sunshine_duration_daily_sum",
    "wind_speed_10m_daily_mean",
    "wind_speed_10m_daily_max",
    "wind_speed_100m_daily_mean",
    "wind_speed_100m_daily_max",
    "wind_gusts_10m_daily_max",
]


def target_columns() -> list[str]:
    return [f"y_{col}" for col in RENEWABLE_TARGET_COLS]


def current_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in RENEWABLE_TARGET_COLS if col in df.columns]


def lag_feature_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for target in RENEWABLE_TARGET_COLS:
        cols.extend([c for c in df.columns if c.startswith(f"lag_") and c.endswith(f"_{target}")])
        cols.extend([c for c in df.columns if c.startswith(f"roll") and c.endswith(f"_{target}")])
    return sorted(cols)


def temporal_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend"]
        if c in df.columns
    ]


def external_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in HOURLY_EXTERNAL_COLUMNS if c in df.columns]


def country_id_columns(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.startswith("country_id_")])


def feature_columns(
    df: pd.DataFrame,
    *,
    include_temporal: bool = True,
    include_external: bool = False,
    include_country_id: bool = False,
) -> list[str]:
    cols = [*current_feature_columns(df), *lag_feature_columns(df)]
    if include_temporal:
        cols.extend(temporal_columns(df))
    if include_external:
        cols.extend(external_columns(df))
    if include_country_id:
        cols.extend(country_id_columns(df))
    return cols


def normalize_data(
    df: pd.DataFrame,
    params: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    if params is None:
        params = {
            "method": "standard",
            "mean": df.mean(),
            "std": df.std().replace(0, 1),
        }
    return (df - params["mean"]) / params["std"], params


def is_generation_column(column: str) -> bool:
    if column == "utc_timestamp" or "Consumption" in column:
        return False
    return True


def matching_generation_columns(columns: list[str], prefix: str) -> list[str]:
    return [
        col
        for col in columns
        if is_generation_column(col)
        and (col == prefix or col.startswith(f"{prefix}_Actual Aggregated"))
    ]


def infer_interval_hours(timestamps: pd.Series) -> pd.Series:
    ts = pd.to_datetime(timestamps, utc=True).sort_values()
    deltas = ts.shift(-1) - ts
    hours = deltas.dt.total_seconds() / 3600.0
    positive = hours[(hours > 0) & np.isfinite(hours)]
    fallback = float(positive.mode().iloc[0]) if not positive.empty else 1.0
    hours = hours.fillna(fallback)
    hours = hours.mask((hours <= 0) | ~np.isfinite(hours), fallback)
    return hours.reindex(ts.index)


def load_generation_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True)
    for col in df.columns:
        if col != "utc_timestamp":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("utc_timestamp").reset_index(drop=True)


def aggregate_country_generation_hourly(df: pd.DataFrame, country_code: str) -> pd.DataFrame:
    df = df.copy().sort_values("utc_timestamp").reset_index(drop=True)
    duration_h = infer_interval_hours(df["utc_timestamp"])
    generation_cols = [c for c in df.columns if is_generation_column(c)]
    energy = df[generation_cols].fillna(0.0).multiply(duration_h.to_numpy(), axis=0)

    out = pd.DataFrame({"utc_timestamp": df["utc_timestamp"]})
    for feature_name, prefixes in RENEWABLE_TECH_PREFIXES.items():
        matches: list[str] = []
        for prefix in prefixes:
            matches.extend(matching_generation_columns(generation_cols, prefix))
        out[feature_name] = energy[matches].sum(axis=1) if matches else 0.0

    total_cols = [c for c in generation_cols if is_generation_column(c)]
    out["total_generation_mwh"] = energy[total_cols].sum(axis=1) if total_cols else 0.0

    out["utc_timestamp"] = out["utc_timestamp"].dt.floor("h")
    hourly = out.groupby("utc_timestamp", as_index=False).sum(numeric_only=True)
    hourly["country_code"] = country_code
    hourly["role"] = role_for_code(country_code)

    hourly["wind_mwh"] = hourly["wind_onshore_mwh"] + hourly["wind_offshore_mwh"]
    hourly["hydro_mwh"] = hourly["hydro_run_of_river_mwh"] + hourly["hydro_reservoir_mwh"]
    hourly["renewable_total_mwh"] = hourly[RENEWABLE_COMPONENT_COLS].sum(axis=1)
    hourly["renewable_share"] = np.where(
        hourly["total_generation_mwh"] > 0,
        hourly["renewable_total_mwh"] / hourly["total_generation_mwh"],
        np.nan,
    )
    return hourly.sort_values("utc_timestamp").reset_index(drop=True)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for country_code, group in df.groupby("country_code", sort=False):
        frame = group.copy()
        local_ts = pd.to_datetime(frame["utc_timestamp"], utc=True).dt.tz_convert(COUNTRY_TIMEZONES[country_code])
        hour = local_ts.dt.hour
        dow = local_ts.dt.dayofweek
        month = local_ts.dt.month - 1
        frame["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
        frame["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
        frame["dow_sin"] = np.sin(2 * np.pi * dow / 7).astype(np.float32)
        frame["dow_cos"] = np.cos(2 * np.pi * dow / 7).astype(np.float32)
        frame["month_sin"] = np.sin(2 * np.pi * month / 12).astype(np.float32)
        frame["month_cos"] = np.cos(2 * np.pi * month / 12).astype(np.float32)
        frame["is_weekend"] = (dow >= 5).astype(np.int8)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True).sort_values(["country_code", "utc_timestamp"]).reset_index(drop=True)


def add_lag_features(
    df: pd.DataFrame,
    *,
    lags: list[int] | None = None,
    roll_windows: list[int] | None = None,
) -> pd.DataFrame:
    lags = lags or DEFAULT_RENEWABLE_LAGS
    roll_windows = roll_windows or DEFAULT_RENEWABLE_ROLL_WINDOWS
    df = df.copy().sort_values(["country_code", "utc_timestamp"]).reset_index(drop=True)
    grouped = df.groupby("country_code", group_keys=False)
    for target in RENEWABLE_TARGET_COLS:
        for lag in lags:
            df[f"lag_{lag}_{target}"] = grouped[target].shift(lag)
        shifted = grouped[target].shift(1)
        for window in roll_windows:
            df[f"roll{window}_mean_{target}"] = shifted.groupby(df["country_code"]).rolling(window).mean().reset_index(level=0, drop=True)
    return df


def add_hour_ahead_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["country_code", "utc_timestamp"]).reset_index(drop=True)
    grouped = df.groupby("country_code", group_keys=False)
    for target in RENEWABLE_TARGET_COLS:
        df[f"y_{target}"] = grouped[target].shift(-1)
    df["target_timestamp"] = grouped["utc_timestamp"].shift(-1)
    return df


def add_country_dummies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dummies = pd.get_dummies(df["country_code"].str.lower(), prefix="country_id").astype(np.int8)
    return pd.concat([df, dummies], axis=1)


def split_by_target_timestamp(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    target_timestamp = pd.to_datetime(df["target_timestamp"], utc=True)
    return {
        "train": df.loc[(target_timestamp >= TRAIN_START) & (target_timestamp <= TRAIN_END)].copy(),
        "val": df.loc[(target_timestamp >= VAL_START) & (target_timestamp <= VAL_END)].copy(),
        "test": df.loc[target_timestamp >= TEST_START].copy(),
    }


def load_all_generation_hourly(generation_dir: Path) -> pd.DataFrame:
    parts = []
    for code in ALL_CODES:
        path = generation_dir / f"entsoe_generation_{code}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        parts.append(aggregate_country_generation_hourly(load_generation_csv(path), code))
    return pd.concat(parts, ignore_index=True).sort_values(["country_code", "utc_timestamp"]).reset_index(drop=True)


# Backward-compatible aliases while the repo finishes moving from D+1 to H+1.
aggregate_country_generation_daily = aggregate_country_generation_hourly
add_day_ahead_targets = add_hour_ahead_targets
split_by_target_date = split_by_target_timestamp
load_all_generation_daily = load_all_generation_hourly
