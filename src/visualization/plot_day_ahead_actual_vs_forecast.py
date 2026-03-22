"""
Plot actual demand vs day-ahead forecasts on a short time window.

The figure is intended to complement the MAE benchmark with a temporal view of
how much each forecast deviates from the realised demand.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import feature_columns, normalize_data, target_columns

DATA_DIR = ROOT / "data" / "processed_long"
FORECAST_DIR = ROOT / "data" / "raw" / "europe" / "forecast"
MODELS_DIR = ROOT / "saved_models"
FIGURES_DIR = ROOT / "docs" / "figures"

RIDGE_PATH = MODELS_DIR / "baseline_ridge.joblib"
XGB_PATH = MODELS_DIR / "baseline_xgb.json"
COUNTRIES = ["ES", "DE", "NL"]
COUNTRY_LABELS = {
    "ES": "Espanya (ENTSO-E millor)",
    "DE": "Alemanya (XGBoost millor)",
    "NL": "Països Baixos (XGBoost millor)",
}
WINDOW_START = pd.Timestamp("2024-01-01", tz="UTC")
WINDOW_END = pd.Timestamp("2024-02-01", tz="UTC")


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


def load_entsoe_forecast(country_code: str) -> pd.DataFrame:
    path = FORECAST_DIR / f"entsoe_load_forecast_{country_code}.csv"
    df = pd.read_csv(path)
    ts_col = "utc_timestamp" if "utc_timestamp" in df.columns else df.columns[0]
    val_col = "forecast" if "forecast" in df.columns else df.columns[-1]
    df = df.rename(columns={ts_col: "delivery_timestamp", val_col: "entsoe_forecast"})
    df["delivery_timestamp"] = pd.to_datetime(df["delivery_timestamp"], utc=True)
    return df[["delivery_timestamp", "entsoe_forecast"]].dropna().sort_values("delivery_timestamp")


def build_delivery_frame(country_df_raw: pd.DataFrame, prediction_mw: np.ndarray, pred_name: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "delivery_timestamp": pd.to_datetime(country_df_raw["utc_timestamp"], utc=True) + pd.Timedelta(hours=24),
            "actual_mw": country_df_raw["y_h24"].to_numpy(dtype=np.float32, copy=True),
            pred_name: prediction_mw,
        }
    )
    return (
        frame.dropna()
        .drop_duplicates(subset=["delivery_timestamp"], keep="last")
        .sort_values("delivery_timestamp")
        .reset_index(drop=True)
    )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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

    y_mean = float(target_params["mean"]["y_h24"])
    y_std = float(target_params["std"]["y_h24"])

    fig, axes = plt.subplots(len(COUNTRIES), 1, figsize=(13, 8.2), sharex=True)

    for ax, code in zip(axes, COUNTRIES):
        raw_domain = test_df_raw[test_df_raw["country_code"] == code].reset_index(drop=True)
        scaled_domain = test_scaled[test_scaled["country_code"] == code].reset_index(drop=True)
        X = scaled_domain[feature_cols].to_numpy(dtype=np.float32, copy=True)

        ridge_pred = ridge.predict(X)[:, 23] * y_std + y_mean
        xgb_pred = xgb.predict(X)[:, 23] * y_std + y_mean

        base = build_delivery_frame(raw_domain, xgb_pred, "xgboost_forecast")
        ridge_frame = build_delivery_frame(raw_domain, ridge_pred, "ridge_forecast")
        entsoe = load_entsoe_forecast(code)

        merged = (
            base.merge(ridge_frame[["delivery_timestamp", "ridge_forecast"]], on="delivery_timestamp", how="left")
            .merge(entsoe, on="delivery_timestamp", how="left")
        )
        merged = merged[
            (merged["delivery_timestamp"] >= WINDOW_START) & (merged["delivery_timestamp"] < WINDOW_END)
        ].copy()

        ax.plot(merged["delivery_timestamp"], merged["actual_mw"], color="#111111", linewidth=1.8, label="Demanda real")
        ax.plot(
            merged["delivery_timestamp"],
            merged["entsoe_forecast"],
            color="#1b9e77",
            linewidth=1.5,
            label="ENTSO-E day-ahead",
        )
        ax.plot(
            merged["delivery_timestamp"],
            merged["xgboost_forecast"],
            color="#1f78b4",
            linewidth=1.5,
            label="XGBoost h=24",
        )
        ax.plot(
            merged["delivery_timestamp"],
            merged["ridge_forecast"],
            color="#7570b3",
            linewidth=1.3,
            alpha=0.9,
            label="Ridge h=24",
        )

        ax.set_title(COUNTRY_LABELS[code], fontsize=11)
        ax.set_ylabel("MW", fontsize=10)
        ax.grid(alpha=0.18)

    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    axes[-1].set_xlabel("Gener 2024", fontsize=10)
    fig.autofmt_xdate(rotation=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Comparativa temporal entre demanda real i prediccions day-ahead", fontsize=15, y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92, bottom=0.10)

    out = FIGURES_DIR / "day_ahead_actual_vs_forecast_jan2024.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
