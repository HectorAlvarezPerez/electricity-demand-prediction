"""
Ablation study: demand lags only vs demand lags + temporal covariates.

Uses the long-format explicit-lag dataset and evaluates zero-shot transfer to ES.
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

from src.data.preprocess import feature_columns, normalize_data, target_columns
from src.paths import FIGURES_DIR, METRICS_DIR, PROCESSED_DATA_DIR, ensure_artifact_dirs

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

DATA_DIR = PROCESSED_DATA_DIR
RESULTS_DIR = METRICS_DIR
RIDGE_ALPHA = 1.0
PRED_LEN = 24


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def to_xy(df: pd.DataFrame, feature_cols: list[str], y_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    Y = df[y_cols].to_numpy(dtype=np.float32, copy=True)
    return X, Y


def scale_frames(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    y_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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


def evaluate_per_domain(predict_fn, test_df: pd.DataFrame, feature_cols: list[str], y_cols: list[str]):
    results = {}
    for code in SOURCE_CODES + [TARGET_CODE]:
        domain_df = test_df[test_df["country_code"] == code].reset_index(drop=True)
        X_test, y_test = to_xy(domain_df, feature_cols, y_cols)
        y_pred = predict_fn(domain_df, X_test)
        results[DOMAIN_RESULT_KEYS[code]] = eval_metrics(y_test, y_pred)
    return results


def run_condition(train_df, val_df, test_df, *, include_temporal: bool):
    tag = "with_features" if include_temporal else "without_features"
    print(f"\n{'=' * 60}")
    print(f"  CONDITION: {tag}")
    print(f"{'=' * 60}")

    y_cols = target_columns(PRED_LEN)
    feature_cols = feature_columns(train_df, include_temporal=include_temporal)
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


def plot_target_ablation(all_results):
    models = ["XGBoost", "Ridge Regression"]
    mae_with = [all_results["with_features"][m]["target_es"]["mae"] for m in models]
    mae_without = [all_results["without_features"][m]["target_es"]["mae"] for m in models]

    y = np.arange(len(models))
    h = 0.3

    fig, ax = plt.subplots(figsize=(10, 4))
    b1 = ax.barh(y + h / 2, mae_with, h, label="Amb temporals", color="#2c7fb8")
    b2 = ax.barh(y - h / 2, mae_without, h, label="Només demanda", color="#fc8d59")

    for bars in (b1, b2):
        for bar in bars:
            ax.text(
                bar.get_width() + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.4f}",
                va="center",
                fontsize=11,
                fontweight="bold",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=13)
    ax.set_xlabel("MAE (Normalitzat) — Target Zero-Shot (ES)", fontsize=12)
    ax.set_title("Ablació: impacte de les covariables temporals", fontsize=14)
    ax.legend(fontsize=11, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.15))
    ax.invert_yaxis()
    fig.tight_layout()

    out = FIGURES_DIR / "ablation_target.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def plot_per_domain_ablation(all_results):
    conditions = {"with_features": "Amb temporals", "without_features": "Només demanda"}
    domains = list(DOMAIN_LABELS.keys())
    labels = [DOMAIN_LABELS[d] for d in domains]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    for ax, (cond_key, cond_title) in zip(axes, conditions.items()):
        models = list(all_results[cond_key].keys())
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        x = np.arange(len(domains))
        w = 0.8 / len(models)
        for i, model in enumerate(models):
            vals = [all_results[cond_key][model][d]["mae"] for d in domains]
            offset = (i - len(models) / 2 + 0.5) * w
            ax.bar(x + offset, vals, w, label=model, color=colors[i])
        ax.set_title(cond_title, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
        ax.set_ylabel("MAE (Normalitzat)", fontsize=11)
        ax.legend(fontsize=9)

    fig.suptitle("MAE per país i model — ablation temporal", fontsize=14, y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / "ablation_per_domain.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def plot_improvement_pct(all_results):
    models = list(all_results["with_features"].keys())
    pct = []
    for model in models:
        mae_with = all_results["with_features"][model]["target_es"]["mae"]
        mae_without = all_results["without_features"][model]["target_es"]["mae"]
        pct.append((mae_without - mae_with) / mae_without * 100)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4c9f70" if p > 0 else "#c0392b" for p in pct]
    bars = ax.bar(models, pct, color=colors)
    for bar, value in zip(bars, pct):
        ax.annotate(
            f"{value:+.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_ylabel("Reducció MAE (%)", fontsize=12)
    ax.set_title("Millora percentual per afegir temporals", fontsize=13)
    ax.axhline(0, color="grey", linewidth=0.8)
    fig.tight_layout()
    out = FIGURES_DIR / "ablation_improvement.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_artifact_dirs()

    print("Loading long-format processed splits...")
    train_df = load_split(DATA_DIR / "train.parquet")
    val_df = load_split(DATA_DIR / "val.parquet")
    test_df = load_split(DATA_DIR / "test.parquet")

    train_df = train_df[train_df["role"] == "source"].reset_index(drop=True)
    val_df = val_df[val_df["role"] == "source"].reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    all_results = {
        "without_features": run_condition(train_df, val_df, test_df, include_temporal=False),
        "with_features": run_condition(train_df, val_df, test_df, include_temporal=True),
    }

    out_json = RESULTS_DIR / "ablation_features.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMetrics saved -> {out_json}")

    plot_target_ablation(all_results)
    plot_per_domain_ablation(all_results)
    plot_improvement_pct(all_results)
    print("\nAblation study complete.")


if __name__ == "__main__":
    main()
