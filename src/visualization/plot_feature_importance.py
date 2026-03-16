"""
Generates XGBoost feature importance plot from a saved model.
Loads from saved_models/baseline_xgb.json (no retraining needed).
Saves to docs/figures/xgb_feature_importance.png
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "docs" / "figures"
MODEL_PATH = ROOT / "saved_models" / "baseline_xgb.json"

TIME_FEATURES_BASE = ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                      "month_sin", "month_cos", "is_weekend"]
SEQ_LEN = 168


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        print(f"ERROR: {MODEL_PATH} not found.")
        print("Run 'python src/models/baselines.py' first to save the model.")
        return

    print(f"Loading XGBoost model from {MODEL_PATH}...")
    xgb = XGBRegressor()
    xgb.load_model(str(MODEL_PATH))

    importances = xgb.feature_importances_

    # Aggregate importance by feature TYPE (sum over all timesteps)
    feat_labels = ["demand"] + TIME_FEATURES_BASE
    num_feat = len(feat_labels)
    aggregated = {}
    for j, feat in enumerate(feat_labels):
        idxs = list(range(j, len(importances), num_feat))
        aggregated[feat] = float(importances[idxs].sum())

    # Sort
    sorted_feats = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
    names = [f[0] for f in sorted_feats]
    values = [f[1] for f in sorted_feats]

    # Nicer labels
    label_map = {
        "demand": "Demanda (lags)",
        "hour_sin": "Hora (sin)", "hour_cos": "Hora (cos)",
        "dow_sin": "Dia setmana (sin)", "dow_cos": "Dia setmana (cos)",
        "month_sin": "Mes (sin)", "month_cos": "Mes (cos)",
        "is_weekend": "Cap de setmana",
    }
    nice_names = [label_map.get(n, n) for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2c7fb8" if n == "demand" else "#41ae76" for n in names]
    bars = ax.barh(nice_names[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel("Importància Acumulada (Gain)", fontsize=12)
    ax.set_title("Importància de Features — XGBoost Baseline", fontsize=14)

    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=10)

    fig.tight_layout()
    out = FIGURES_DIR / "xgb_feature_importance.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved → {out}")

    # Print breakdown
    total = sum(values)
    print("\nFeature importance breakdown:")
    for n, v in sorted_feats:
        print(f"  {label_map.get(n,n):25s}  {v:.4f}  ({v/total*100:.1f}%)")


if __name__ == "__main__":
    main()
