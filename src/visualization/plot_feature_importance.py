"""
Generate XGBoost feature-importance plots for the long-format baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "docs" / "figures"
MODEL_PATH = ROOT / "saved_models" / "baseline_xgb.json"
MODEL_META_PATH = ROOT / "saved_models" / "baseline_xgb_features.json"


def feature_family(col_name: str) -> str:
    if col_name == "demand" or col_name.startswith("lag_"):
        return "demand"
    if col_name.startswith("hour_"):
        return "hour"
    if col_name.startswith("dow_"):
        return "dow"
    if col_name.startswith("month_"):
        return "month"
    if col_name == "is_weekend":
        return "weekend"
    if col_name in {"temperature_2m", "temp_daily_mean", "temp_daily_max", "temp_daily_min"}:
        return col_name
    if col_name.startswith("country_id_"):
        return "country_id"
    return "other"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        print(f"ERROR: {MODEL_PATH} not found.")
        print("Run 'python src/models/baselines.py' first to save the model.")
        return
    if not MODEL_META_PATH.exists():
        print(f"ERROR: {MODEL_META_PATH} not found.")
        print("Run 'python src/models/baselines.py' first to save the feature metadata.")
        return

    with open(MODEL_META_PATH) as f:
        meta = json.load(f)
    feature_names = meta["feature_cols"]

    print(f"Loading XGBoost model from {MODEL_PATH}...")
    xgb = XGBRegressor()
    xgb.load_model(str(MODEL_PATH))

    importances = xgb.feature_importances_
    if len(importances) != len(feature_names):
        raise ValueError(
            f"Feature importance length ({len(importances)}) does not match "
            f"stored feature count ({len(feature_names)})."
        )

    aggregated = {}
    for name, value in zip(feature_names, importances):
        family = feature_family(name)
        aggregated.setdefault(family, 0.0)
        aggregated[family] += float(value)

    sorted_feats = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
    names = [f[0] for f in sorted_feats]
    values = [f[1] for f in sorted_feats]

    label_map = {
        "demand": "Demanda i lags",
        "hour": "Hora local",
        "dow": "Dia setmana",
        "month": "Mes",
        "weekend": "Cap de setmana",
        "temperature_2m": "Temperatura horària",
        "temp_daily_mean": "Temperatura diària mitjana",
        "temp_daily_max": "Temperatura diària màxima",
        "temp_daily_min": "Temperatura diària mínima",
        "country_id": "Country ID",
        "other": "Altres",
    }
    nice_names = [label_map.get(name, name) for name in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [
        "#2c7fb8" if name == "demand"
        else "#41ae76" if name in {"hour", "dow", "month", "weekend"}
        else "#fd8d3c" if "temp" in name
        else "#756bb1"
        for name in names
    ]
    bars = ax.barh(nice_names[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel("Importància acumulada", fontsize=12)
    ax.set_title("Importància de features — XGBoost long-format", fontsize=14)

    for bar, val in zip(bars, values[::-1]):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=10,
        )

    fig.tight_layout()
    out = FIGURES_DIR / "xgb_feature_importance.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")

    total = sum(values)
    print("\nFeature importance breakdown:")
    for name, value in sorted_feats:
        print(f"  {label_map.get(name, name):25s}  {value:.4f}  ({value / total * 100:.1f}%)")


if __name__ == "__main__":
    main()
