"""Shared helpers for the daily renewables forecasting dataset."""
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
    VAL_END,
    VAL_START,
    role_for_code,
)


RENEWABLE_TARGET_COLS = [
    "solar_mwh",
    "wind_mwh",
    "hydro_mwh",
    "renewable_total_mwh",
    "renewable_share",
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

DEFAULT_RENEWABLE_LAGS = [1, 2, 7, 14, 30]
DEFAULT_RENEWABLE_ROLL_WINDOWS = [7, 14, 30]

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
    return [c for c in ["dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend"] if c in df.columns]


def external_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in DAILY_EXTERNAL_COLUMNS if c in df.columns]


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


def aggregate_country_generation_daily(df: pd.DataFrame, country_code: str) -> pd.DataFrame:
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

    local_tz = COUNTRY_TIMEZONES[country_code]
    out["date"] = out["utc_timestamp"].dt.tz_convert(local_tz).dt.date
    daily = out.drop(columns=["utc_timestamp"]).groupby("date", as_index=False).sum(numeric_only=True)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["country_code"] = country_code
    daily["role"] = role_for_code(country_code)

    daily["wind_mwh"] = daily["wind_onshore_mwh"] + daily["wind_offshore_mwh"]
    daily["hydro_mwh"] = daily["hydro_run_of_river_mwh"] + daily["hydro_reservoir_mwh"]
    daily["renewable_total_mwh"] = daily[RENEWABLE_COMPONENT_COLS].sum(axis=1)
    daily["renewable_share"] = np.where(
        daily["total_generation_mwh"] > 0,
        daily["renewable_total_mwh"] / daily["total_generation_mwh"],
        np.nan,
    )
    return daily.sort_values("date").reset_index(drop=True)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date = pd.to_datetime(df["date"])
    dow = date.dt.dayofweek
    month = date.dt.month - 1
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7).astype(np.float32)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7).astype(np.float32)
    df["month_sin"] = np.sin(2 * np.pi * month / 12).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * month / 12).astype(np.float32)
    df["is_weekend"] = (dow >= 5).astype(np.int8)
    return df


def add_lag_features(
    df: pd.DataFrame,
    *,
    lags: list[int] | None = None,
    roll_windows: list[int] | None = None,
) -> pd.DataFrame:
    lags = lags or DEFAULT_RENEWABLE_LAGS
    roll_windows = roll_windows or DEFAULT_RENEWABLE_ROLL_WINDOWS
    df = df.copy().sort_values(["country_code", "date"]).reset_index(drop=True)
    grouped = df.groupby("country_code", group_keys=False)
    for target in RENEWABLE_TARGET_COLS:
        for lag in lags:
            df[f"lag_{lag}_{target}"] = grouped[target].shift(lag)
        shifted = grouped[target].shift(1)
        for window in roll_windows:
            df[f"roll{window}_mean_{target}"] = shifted.groupby(df["country_code"]).rolling(window).mean().reset_index(level=0, drop=True)
    return df


def add_day_ahead_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["country_code", "date"]).reset_index(drop=True)
    grouped = df.groupby("country_code", group_keys=False)
    for target in RENEWABLE_TARGET_COLS:
        df[f"y_{target}"] = grouped[target].shift(-1)
    df["target_date"] = grouped["date"].shift(-1)
    return df


def add_country_dummies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dummies = pd.get_dummies(df["country_code"].str.lower(), prefix="country_id").astype(np.int8)
    return pd.concat([df, dummies], axis=1)


def split_by_target_date(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    target_date = pd.to_datetime(df["target_date"], utc=True)
    return {
        "train": df.loc[(target_date >= TRAIN_START) & (target_date <= TRAIN_END)].copy(),
        "val": df.loc[(target_date >= VAL_START) & (target_date <= VAL_END)].copy(),
        "test": df.loc[target_date >= pd.Timestamp("2024-01-01", tz="UTC")].copy(),
    }


def load_all_generation_daily(generation_dir: Path) -> pd.DataFrame:
    parts = []
    for code in ALL_CODES:
        path = generation_dir / f"entsoe_generation_{code}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        parts.append(aggregate_country_generation_daily(load_generation_csv(path), code))
    return pd.concat(parts, ignore_index=True).sort_values(["country_code", "date"]).reset_index(drop=True)

