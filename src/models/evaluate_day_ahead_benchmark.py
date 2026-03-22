"""
Evaluate day-ahead (h=24) forecasts per country.

Benchmarks included:
  - ENTSO-E published day-ahead load forecast
  - Daily Naive
  - Ridge Regression
  - XGBoost

The script maps every model prediction to its delivery timestamp and computes
MAE/RMSE in MW so the benchmark is comparable across all domains.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import feature_columns, normalize_data, target_columns
from src.paths import METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_artifact_dirs

TARGET_CODE = "ES"
SOURCE_CODES = ["BE", "DE", "FR", "GR", "IT", "NL", "PT"]
ALL_CODES = [TARGET_CODE, *SOURCE_CODES]

DATA_DIR = PROCESSED_DATA_DIR
FORECAST_DIR = RAW_DATA_DIR / "europe" / "forecast"

RIDGE_PATH = MODELS_DIR / "baseline_ridge.joblib"
XGB_PATH = MODELS_DIR / "baseline_xgb.json"
XGB_META_PATH = MODELS_DIR / "baseline_xgb_features.json"


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"{name}.parquet")


def scale_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    y_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train = train_df.copy()
    test = test_df.copy()

    train_features, feature_params = normalize_data(train[feature_cols], method="standard")
    test_features, _ = normalize_data(test[feature_cols], method="standard", params=feature_params)

    train_targets, target_params = normalize_data(train[y_cols], method="standard")
    test_targets, _ = normalize_data(test[y_cols], method="standard", params=target_params)

    train[feature_cols] = train_features
    test[feature_cols] = test_features
    train[y_cols] = train_targets
    test[y_cols] = test_targets
    return train, test, target_params


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae_mw": float(mean_absolute_error(y_true, y_pred)),
        "rmse_mw": float(root_mean_squared_error(y_true, y_pred)),
    }


def predict_daily_naive(df: pd.DataFrame) -> np.ndarray:
    return df["demand"].to_numpy(dtype=np.float32, copy=True)


def build_day_ahead_delivery_frame(
    country_df_raw: pd.DataFrame,
    y_true_h24: np.ndarray,
    y_pred_h24: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "delivery_timestamp": pd.to_datetime(country_df_raw["utc_timestamp"], utc=True) + pd.Timedelta(hours=24),
            "actual_mw": y_true_h24,
            "pred_mw": y_pred_h24,
        }
    )
    frame = frame.dropna().drop_duplicates(subset=["delivery_timestamp"], keep="last")
    return frame.sort_values("delivery_timestamp").reset_index(drop=True)


def load_entsoe_forecast(country_code: str) -> pd.DataFrame:
    path = FORECAST_DIR / f"entsoe_load_forecast_{country_code}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src/data/download_entsoe_forecast.py first."
        )

    df = pd.read_csv(path)
    ts_col = "utc_timestamp" if "utc_timestamp" in df.columns else df.columns[0]
    val_col = "forecast" if "forecast" in df.columns else df.columns[-1]
    df = df.rename(columns={ts_col: "delivery_timestamp", val_col: "forecast_mw"})
    df["delivery_timestamp"] = pd.to_datetime(df["delivery_timestamp"], utc=True)
    return df[["delivery_timestamp", "forecast_mw"]].dropna().sort_values("delivery_timestamp")


def evaluate_entsoe_forecast(country_df_raw: pd.DataFrame) -> dict[str, float]:
    actual = (
        pd.DataFrame(
            {
                "delivery_timestamp": pd.to_datetime(country_df_raw["utc_timestamp"], utc=True) + pd.Timedelta(hours=24),
                "actual_mw": country_df_raw["y_h24"].to_numpy(dtype=np.float32, copy=True),
            }
        )
        .dropna()
        .drop_duplicates(subset=["delivery_timestamp"], keep="last")
        .sort_values("delivery_timestamp")
    )
    forecast = load_entsoe_forecast(country_df_raw["country_code"].iloc[0])
    merged = actual.merge(forecast, on="delivery_timestamp", how="inner")
    metrics = eval_metrics(merged["actual_mw"].to_numpy(), merged["forecast_mw"].to_numpy())
    metrics["n_hours"] = int(len(merged))
    return metrics


def main() -> None:
    ensure_artifact_dirs()

    train_df = load_split("train")
    test_df_raw = load_split("test")

    train_df = train_df[train_df["role"] == "source"].reset_index(drop=True)

    y_cols = target_columns(24)
    feature_cols = feature_columns(
        train_df,
        include_temporal=True,
        include_weather=True,
        include_country_id=True,
    )

    train_scaled, test_scaled, target_params = scale_frames(train_df, test_df_raw, feature_cols, y_cols)

    ridge = joblib.load(RIDGE_PATH)
    xgb = XGBRegressor()
    xgb.load_model(str(XGB_PATH))

    x_mean = float(target_params["mean"]["y_h24"])
    x_std = float(target_params["std"]["y_h24"])

    results: dict[str, dict[str, dict[str, float]]] = {}
    for code in SOURCE_CODES + [TARGET_CODE]:
        domain_key = f"source_{code.lower()}" if code != TARGET_CODE else "target_es"
        raw_domain = test_df_raw[test_df_raw["country_code"] == code].reset_index(drop=True)
        scaled_domain = test_scaled[test_scaled["country_code"] == code].reset_index(drop=True)

        X = scaled_domain[feature_cols].to_numpy(dtype=np.float32, copy=True)

        ridge_pred_scaled = ridge.predict(X)[:, 23]
        xgb_pred_scaled = xgb.predict(X)[:, 23]
        y_true_h24 = raw_domain["y_h24"].to_numpy(dtype=np.float32, copy=True)

        ridge_pred_mw = ridge_pred_scaled * x_std + x_mean
        xgb_pred_mw = xgb_pred_scaled * x_std + x_mean
        naive_pred_mw = predict_daily_naive(raw_domain)

        ridge_frame = build_day_ahead_delivery_frame(raw_domain, y_true_h24, ridge_pred_mw)
        xgb_frame = build_day_ahead_delivery_frame(raw_domain, y_true_h24, xgb_pred_mw)
        naive_frame = build_day_ahead_delivery_frame(raw_domain, y_true_h24, naive_pred_mw)

        results[domain_key] = {
            "entsoe_day_ahead": evaluate_entsoe_forecast(raw_domain),
            "daily_naive_h24": eval_metrics(naive_frame["actual_mw"].to_numpy(), naive_frame["pred_mw"].to_numpy()),
            "ridge_h24": eval_metrics(ridge_frame["actual_mw"].to_numpy(), ridge_frame["pred_mw"].to_numpy()),
            "xgboost_h24": eval_metrics(xgb_frame["actual_mw"].to_numpy(), xgb_frame["pred_mw"].to_numpy()),
        }

    out_path = METRICS_DIR / "day_ahead_benchmark_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved -> {out_path}")
    for domain_key, metrics in results.items():
        print(
            domain_key,
            "| ENTSOE MAE:", round(metrics["entsoe_day_ahead"]["mae_mw"], 2),
            "| XGB h24 MAE:", round(metrics["xgboost_h24"]["mae_mw"], 2),
        )


if __name__ == "__main__":
    main()
