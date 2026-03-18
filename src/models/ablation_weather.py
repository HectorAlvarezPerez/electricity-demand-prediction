"""
Ablation study over temporal and weather covariates.

Evaluates four configurations on the long-format explicit-lag dataset:
    - demand_only
    - temporal_only = demand + temporal covariates
    - weather_only = demand + weather covariates
    - all_features = demand + temporal covariates + weather covariates

The target domain is evaluated zero-shot on ES.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import feature_columns, normalize_data, target_columns, weather_columns

TARGET_CODE = "ES"
SOURCE_CODES = ["BE", "DE", "FR", "GR", "IT", "NL", "PT"]
DOMAIN_LABELS = {
    "source_be": "Bèlgica",
    "source_de": "Alemanya",
    "source_fr": "França",
    "source_gr": "Grècia",
    "source_it": "Itàlia",
    "source_nl": "P. Baixos",
    "source_pt": "Portugal",
    "target_es": "Espanya (Target)",
}
DOMAIN_RESULT_KEYS = {
    "BE": "source_be",
    "DE": "source_de",
    "FR": "source_fr",
    "GR": "source_gr",
    "IT": "source_it",
    "NL": "source_nl",
    "PT": "source_pt",
    "ES": "target_es",
}

DATA_DIR = ROOT / "data" / "processed_long"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "docs" / "figures"
RIDGE_ALPHA = 1.0
PRED_LEN = 24

CONDITION_LABELS = {
    "demand_only": "Només demanda",
    "temporal_only": "Demanda + temporals",
    "weather_only": "Demanda + meteo",
    "all_features": "Demanda + temporals + meteo",
}
CONDITION_COLORS = {
    "demand_only": "#fc8d59",
    "temporal_only": "#2c7fb8",
    "weather_only": "#756bb1",
    "all_features": "#41ae76",
}


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def to_xy(df: pd.DataFrame, feature_cols: list[str], y_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    Y = df[y_cols].to_numpy(dtype=np.float32, copy=True)
    return X, Y


def scale_frames(train_df, val_df, test_df, feature_cols, y_cols):
    train = train_df.copy()
    val = val_df.copy()
    test = test_df.copy()

    train_features, feature_params = normalize_data(train[feature_cols], method="standard")
    val_features, _ = normalize_data(val[feature_cols], method="standard", params=feature_params)
    test_features, _ = normalize_data(test[feature_cols], method="standard", params=feature_params)

    train_targets, target_params = normalize_data(train[y_cols], method="standard")
    val_targets, _ = normalize_data(val[y_cols], method="standard", params=target_params)
    test_targets, _ = normalize_data(test[y_cols], method="standard", params=target_params)

    train[feature_cols] = train_features
    val[feature_cols] = val_features
    test[feature_cols] = test_features
    train[y_cols] = train_targets
    val[y_cols] = val_targets
    test[y_cols] = test_targets
    return train, val, test


def eval_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
    }


def predict_daily_naive(df: pd.DataFrame) -> np.ndarray:
    preds = []
    for horizon in range(1, PRED_LEN + 1):
        source_lag = 24 - horizon
        if source_lag == 0:
            preds.append(df["demand"].to_numpy(dtype=np.float32))
        else:
            preds.append(df[f"lag_{source_lag}"].to_numpy(dtype=np.float32))
    return np.stack(preds, axis=1)


def evaluate_per_domain(predict_fn, test_df, feature_cols, y_cols):
    results = {}
    for code in SOURCE_CODES + [TARGET_CODE]:
        domain_df = test_df[test_df["country_code"] == code].reset_index(drop=True)
        X_test, y_test = to_xy(domain_df, feature_cols, y_cols)
        y_pred = predict_fn(domain_df, X_test)
        results[DOMAIN_RESULT_KEYS[code]] = eval_metrics(y_test, y_pred)
    return results


def feature_cols_for_mode(df: pd.DataFrame, mode: str) -> list[str]:
    if mode == "demand_only":
        return feature_columns(df, include_temporal=False, include_weather=False)
    if mode == "temporal_only":
        return feature_columns(df, include_temporal=True, include_weather=False)
    if mode == "weather_only":
        return feature_columns(df, include_temporal=False, include_weather=True)
    if mode == "all_features":
        return feature_columns(df, include_temporal=True, include_weather=True)
    raise ValueError(f"Unknown mode: {mode}")


def run_condition(train_df, val_df, test_df, mode: str):
    print(f"\n{'=' * 60}")
    print(f"  CONDITION: {mode}")
    print(f"{'=' * 60}")

    y_cols = target_columns(PRED_LEN)
    feature_cols = feature_cols_for_mode(train_df, mode)
    train_scaled, val_scaled, test_scaled = scale_frames(train_df, val_df, test_df, feature_cols, y_cols)
    X_train, y_train = to_xy(train_scaled, feature_cols, y_cols)
    X_val, y_val = to_xy(val_scaled, feature_cols, y_cols)

    results = {}
    results["Daily Naive"] = evaluate_per_domain(
        lambda domain_df, _X: predict_daily_naive(domain_df),
        test_scaled,
        feature_cols,
        y_cols,
    )

    ridge = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(X_train, y_train)
    results["Ridge Regression"] = evaluate_per_domain(
        lambda _df, X: ridge.predict(X),
        test_scaled,
        feature_cols,
        y_cols,
    )

    xgb = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
    )
    xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=10,
    )
    results["XGBoost"] = evaluate_per_domain(
        lambda _df, X: xgb.predict(X),
        test_scaled,
        feature_cols,
        y_cols,
    )
    return results


def plot_target_pair(all_results, left_cond: str, right_cond: str, out_name: str, title: str):
    models = ["XGBoost", "Ridge Regression"]
    conditions = [left_cond, right_cond]
    y = np.arange(len(models))
    h = 0.28

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, cond in enumerate(conditions):
        vals = [all_results[cond][m]["target_es"]["mae"] for m in models]
        offset = (i - 0.5) * h
        bars = ax.barh(
            y + offset,
            vals,
            h,
            label=CONDITION_LABELS[cond],
            color=CONDITION_COLORS[cond],
        )
        for bar in bars:
            ax.text(
                bar.get_width() + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.4f}",
                va="center",
                fontsize=10,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=13)
    ax.set_xlabel("MAE (Normalitzat) — Target Zero-Shot (ES)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.15))
    ax.invert_yaxis()
    fig.tight_layout()

    out = FIGURES_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def plot_target_all(all_results):
    models = ["XGBoost", "Ridge Regression"]
    conditions = ["demand_only", "temporal_only", "weather_only", "all_features"]
    y = np.arange(len(models))
    h = 0.18

    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, cond in enumerate(conditions):
        vals = [all_results[cond][m]["target_es"]["mae"] for m in models]
        offset = (i - 1.5) * h
        bars = ax.barh(
            y + offset,
            vals,
            h,
            label=CONDITION_LABELS[cond],
            color=CONDITION_COLORS[cond],
        )
        for bar in bars:
            ax.text(
                bar.get_width() + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.4f}",
                va="center",
                fontsize=9,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=13)
    ax.set_xlabel("MAE (Normalitzat) — Target Zero-Shot (ES)", fontsize=12)
    ax.set_title("Comparativa global de variables exògenes", fontsize=14)
    ax.legend(fontsize=9, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.18))
    ax.invert_yaxis()
    fig.tight_layout()

    out = FIGURES_DIR / "ablation_weather_all_target.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def plot_per_domain_all(all_results):
    domains = list(DOMAIN_LABELS.keys())
    labels = [DOMAIN_LABELS[d] for d in domains]
    models = ["Ridge Regression", "XGBoost"]
    conditions = ["demand_only", "temporal_only", "weather_only", "all_features"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    for ax, model in zip(axes, models):
        x = np.arange(len(domains))
        w = 0.18
        for i, cond in enumerate(conditions):
            vals = [all_results[cond][model][d]["mae"] for d in domains]
            offset = (i - 1.5) * w
            ax.bar(x + offset, vals, w, label=CONDITION_LABELS[cond], color=CONDITION_COLORS[cond])
        ax.set_title(model, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
        ax.set_ylabel("MAE (Normalitzat)", fontsize=11)
        ax.legend(fontsize=9)

    fig.suptitle("MAE per país segons conjunt de variables", fontsize=14, y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / "ablation_weather_all_per_domain.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading long-format processed splits...")
    train_df = load_split(DATA_DIR / "train.parquet")
    val_df = load_split(DATA_DIR / "val.parquet")
    test_df = load_split(DATA_DIR / "test.parquet")

    if not weather_columns(train_df):
        raise ValueError(
            "No weather columns found in data/processed_long. "
            "Regenerate the processed dataset with raw weather files available."
        )

    train_df = train_df[train_df["role"] == "source"].reset_index(drop=True)
    val_df = val_df[val_df["role"] == "source"].reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    all_results = {
        "demand_only": run_condition(train_df, val_df, test_df, "demand_only"),
        "temporal_only": run_condition(train_df, val_df, test_df, "temporal_only"),
        "weather_only": run_condition(train_df, val_df, test_df, "weather_only"),
        "all_features": run_condition(train_df, val_df, test_df, "all_features"),
    }

    out_json = RESULTS_DIR / "ablation_weather.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMetrics saved -> {out_json}")

    plot_target_pair(
        all_results,
        "demand_only",
        "temporal_only",
        "ablation_weather_temporal_target.png",
        "Impacte de les variables temporals",
    )
    plot_target_pair(
        all_results,
        "demand_only",
        "weather_only",
        "ablation_weather_meteo_target.png",
        "Impacte de les variables meteorològiques",
    )
    plot_target_all(all_results)
    plot_per_domain_all(all_results)
    print("\nWeather ablation complete.")


if __name__ == "__main__":
    main()
