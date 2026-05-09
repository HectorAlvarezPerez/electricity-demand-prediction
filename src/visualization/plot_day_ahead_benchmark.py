"""
Plot one grouped-bar chart comparing actual vs predicted next-day demand.

For each country, the chart shows average delivered demand on the aligned
evaluation timestamps for:
  - actual demand
  - ENTSO-E previous-day forecast
  - Ridge h=24
  - XGBoost h=24
"""
from __future__ import annotations

from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import feature_columns, normalize_data, target_columns
from src.paths import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_artifact_dirs
from src.visualization.plot_style import apply_report_bar_style, color_for_model, color_for_series

DATA_DIR = PROCESSED_DATA_DIR
FORECAST_DIR = RAW_DATA_DIR / "europe" / "forecast"

RIDGE_PATH = MODELS_DIR / "baseline_ridge.joblib"
XGB_PATH = MODELS_DIR / "baseline_xgb.json"

DOMAIN_ORDER = [
    ("source_be", "BE", "Bèlgica"),
    ("source_de", "DE", "Alemanya"),
    ("source_fr", "FR", "França"),
    ("source_gr", "GR", "Grècia"),
    ("source_it", "IT", "Itàlia"),
    ("source_nl", "NL", "Països Baixos"),
    ("source_pt", "PT", "Portugal"),
    ("target_es", "ES", "Espanya (Target)"),
]


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


def build_delivery_frame(
    country_df_raw: pd.DataFrame,
    prediction_mw: np.ndarray,
    pred_name: str,
) -> pd.DataFrame:
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

    y_mean = float(target_params["mean"]["y_h24"])
    y_std = float(target_params["std"]["y_h24"])

    labels: list[str] = []
    actual_vals: list[float] = []
    entsoe_vals: list[float] = []
    ridge_vals: list[float] = []
    xgb_vals: list[float] = []

    for _, code, label in DOMAIN_ORDER:
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
            .merge(entsoe, on="delivery_timestamp", how="inner")
        )

        labels.append(label)
        actual_vals.append(float(merged["actual_mw"].mean()))
        entsoe_vals.append(float(merged["entsoe_forecast"].mean()))
        ridge_vals.append(float(merged["ridge_forecast"].mean()))
        xgb_vals.append(float(merged["xgboost_forecast"].mean()))

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.bar(x - 1.5 * width, actual_vals, width=width, color=color_for_series("Demanda real"), label="Demanda real")
    ax.bar(x - 0.5 * width, entsoe_vals, width=width, color=color_for_series("ENTSO-E dia anterior"), label="ENTSO-E dia anterior")
    ax.bar(x + 0.5 * width, ridge_vals, width=width, color=color_for_model("Ridge h=24"), label="Ridge h=24")
    ax.bar(x + 1.5 * width, xgb_vals, width=width, color=color_for_model("XGBoost"), label="XGBoost h=24")

    ax.set_ylabel("Demanda mitjana (MW)", fontsize=12)
    ax.set_title("Comparativa de previsió del dia anterior de demanda mitjana per país", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    apply_report_bar_style(ax, grid_alpha=0.2)
    ax.legend(fontsize=10, ncol=2)
    fig.tight_layout()

    out = FIGURES_DIR / "day_ahead_benchmark_per_domain.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
