"""
Ablation study: WITH vs WITHOUT temporal features (sin/cos/is_weekend).

Trains Daily Naive, Linear Regression, and XGBoost under both conditions,
saves metrics to results/ablation_features.json,
saves plots to docs/figures/ablation_*.png.

Run:
    python src/models/ablation_features.py
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from src.data.dataset import ElectricityDemandDataset
from src.data.preprocess import normalize_data

# ── constants ───────────────────────────────────────────────────────────────────
SOURCE_CODES = ["source_be", "source_de", "source_fr", "source_gr",
                "source_it", "source_nl", "source_pt"]
TARGET_CODE = "target_es"
TIME_FEATURES_BASE = ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                      "month_sin", "month_cos", "is_weekend"]

DOMAIN_LABELS = {
    "source_be": "Bèlgica", "source_de": "Alemanya", "source_fr": "França",
    "source_gr": "Grècia", "source_it": "Itàlia", "source_nl": "P. Baixos",
    "source_pt": "Portugal", "target_es": "Espanya (Target)",
}

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "docs" / "figures"

SEQ_LEN = 168
PRED_LEN = 24

# ── data helpers ────────────────────────────────────────────────────────────────

def prepare_domain_data(df, country_col, use_time_features, seq_len=SEQ_LEN, pred_len=PRED_LEN):
    """Build (X, Y) matrices for one country, with or without time features."""
    if use_time_features:
        prefix = country_col.split("_")[-1]
        time_cols = [f"{prefix}_{f}" for f in TIME_FEATURES_BASE]
        cols = [country_col] + time_cols
    else:
        cols = [country_col]

    df_country = df[cols]
    dataset = ElectricityDemandDataset(df_country, seq_len=seq_len,
                                       pred_len=pred_len, target_col=country_col)

    n = len(dataset)
    X = np.empty((n, seq_len * len(cols)), dtype=np.float32)
    Y = np.empty((n, pred_len), dtype=np.float32)
    for i in tqdm(range(n), desc=f"  {country_col} ({'feat' if use_time_features else 'raw'})",
                  leave=False):
        x, y = dataset[i]
        X[i] = x.numpy().flatten()
        Y[i] = y.numpy().flatten()
    return X, Y


def build_multidomain(df, cols, use_time_features):
    Xs, Ys = [], []
    for c in cols:
        X, Y = prepare_domain_data(df, c, use_time_features)
        Xs.append(X); Ys.append(Y)
    return np.vstack(Xs), np.vstack(Ys)


def eval_metrics(Y_true, Y_pred):
    return {
        "mae": float(mean_absolute_error(Y_true, Y_pred)),
        "rmse": float(root_mean_squared_error(Y_true, Y_pred)),
    }


def evaluate_model(name, predict_fn, test_df, use_time_features):
    """Returns dict  {domain_code: {mae, rmse}}."""
    results = {}
    for code in tqdm(SOURCE_CODES, desc=f"  Eval {name}"):
        Xt, Yt = prepare_domain_data(test_df, code, use_time_features)
        results[code] = eval_metrics(Yt, predict_fn(Xt, code))

    Xt, Yt = prepare_domain_data(test_df, TARGET_CODE, use_time_features)
    results[TARGET_CODE] = eval_metrics(Yt, predict_fn(Xt, TARGET_CODE))
    return results


# ── plotting helpers ────────────────────────────────────────────────────────────

def plot_target_ablation(all_results):
    """Paired horizontal bar chart: with vs without features for LR and XGBoost.
    Daily Naive is excluded since it is identical in both conditions."""
    models = ["XGBoost", "Linear Regression"]
    mae_with    = [all_results["with_features"][m][TARGET_CODE]["mae"] for m in models]
    mae_without = [all_results["without_features"][m][TARGET_CODE]["mae"] for m in models]

    y = np.arange(len(models))
    h = 0.3

    fig, ax = plt.subplots(figsize=(10, 4))

    b1 = ax.barh(y + h/2, mae_with, h, label="Amb features temporals",
                 color="#2c7fb8", edgecolor="white", linewidth=0.5)
    b2 = ax.barh(y - h/2, mae_without, h, label="Sense features temporals",
                 color="#fc8d59", edgecolor="white", linewidth=0.5)

    # Value labels on bars
    for bar in b1:
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.4f}", va="center", fontsize=11, fontweight="bold",
                color="#2c7fb8")
    for bar in b2:
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.4f}", va="center", fontsize=11, fontweight="bold",
                color="#fc8d59")

    # Show absolute MAE delta between pairs
    for i in range(len(models)):
        delta = mae_without[i] - mae_with[i]
        mid_x = max(mae_with[i], mae_without[i]) + 0.04
        ax.annotate(f"Δ +{delta:.3f}",
                    xy=(mid_x, y[i]),
                    fontsize=11, fontweight="bold", color="#c0392b",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef0ef",
                              edgecolor="#c0392b", linewidth=1.2))

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=13)
    ax.set_xlabel("MAE (Normalitzat) — Target Zero-Shot (ES)", fontsize=12)
    ax.set_title("Ablació: Impacte de les Features Temporals", fontsize=14)
    ax.legend(fontsize=11, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.15))
    ax.set_xlim(0, max(mae_without) + 0.09)
    ax.invert_yaxis()
    fig.tight_layout()

    out = FIGURES_DIR / "ablation_target.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved → {out}")


def plot_per_domain_ablation(all_results):
    """Per-domain MAE grouped by model, one subplot per condition."""
    conditions = {"with_features": "Amb Features Temporals",
                  "without_features": "Sense Features Temporals"}
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
            offset = (i - len(models)/2 + 0.5) * w
            ax.bar(x + offset, vals, w, label=model, color=colors[i])
        ax.set_title(cond_title, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
        ax.set_ylabel("MAE (Normalitzat)", fontsize=11)
        ax.legend(fontsize=9)

    fig.suptitle("MAE per País i Model — Ablació de Features Temporals", fontsize=14, y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / "ablation_per_domain.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved → {out}")


def plot_improvement_pct(all_results):
    """Bar chart showing % MAE improvement from adding temporal features, per model."""
    models = list(all_results["with_features"].keys())
    pct = []
    for m in models:
        mae_w  = all_results["with_features"][m][TARGET_CODE]["mae"]
        mae_wo = all_results["without_features"][m][TARGET_CODE]["mae"]
        pct.append((mae_wo - mae_w) / mae_wo * 100)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4c9f70" if p > 0 else "#c0392b" for p in pct]
    bars = ax.bar(models, pct, color=colors)
    for bar, p in zip(bars, pct):
        ax.annotate(f"{p:+.1f}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylabel("Reducció MAE (%)", fontsize=12)
    ax.set_title("Millora (%) per afegir Features Temporals — Target Zero-Shot",
                 fontsize=13)
    ax.axhline(0, color="grey", linewidth=0.8)
    fig.tight_layout()

    out = FIGURES_DIR / "ablation_improvement.png"
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"Saved → {out}")


# ── main ────────────────────────────────────────────────────────────────────────

def run_condition(train_df, val_df, test_df, use_time_features):
    tag = "WITH" if use_time_features else "WITHOUT"
    print(f"\n{'='*60}")
    print(f"  CONDITION: {tag} temporal features")
    print(f"{'='*60}")

    print("Preparing training data...")
    X_train, Y_train = build_multidomain(train_df, SOURCE_CODES, use_time_features)
    print(f"  Train shapes — X: {X_train.shape}, Y: {Y_train.shape}")

    print("Preparing validation data...")
    X_val, Y_val = build_multidomain(val_df, SOURCE_CODES, use_time_features)

    results = {}

    # Daily Naive
    print(f"\n[{tag}] Evaluating Daily Naive...")
    num_features = 1 + len(TIME_FEATURES_BASE) if use_time_features else 1
    def predict_naive(X, _col):
        X_r = X.reshape(-1, SEQ_LEN, num_features)
        return X_r[:, -PRED_LEN:, 0]
    results["Daily Naive"] = evaluate_model("Daily Naive", predict_naive,
                                            test_df, use_time_features)
    print(f"  Target MAE: {results['Daily Naive'][TARGET_CODE]['mae']:.4f}")

    # Linear Regression
    print(f"\n[{tag}] Training Linear Regression...")
    lr = LinearRegression(n_jobs=-1)
    lr.fit(X_train, Y_train)
    results["Linear Regression"] = evaluate_model(
        "Linear Regression", lambda X, _: lr.predict(X), test_df, use_time_features)
    print(f"  Target MAE: {results['Linear Regression'][TARGET_CODE]['mae']:.4f}")

    # XGBoost
    print(f"\n[{tag}] Training XGBoost...")
    xgb = XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.1,
                        tree_method="hist", n_jobs=-1)
    xgb.fit(X_train, Y_train,
            eval_set=[(X_train, Y_train), (X_val, Y_val)],
            verbose=10)
    results["XGBoost"] = evaluate_model(
        "XGBoost", lambda X, _: xgb.predict(X), test_df, use_time_features)
    print(f"  Target MAE: {results['XGBoost'][TARGET_CODE]['mae']:.4f}")

    return results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    val_df   = pd.read_parquet(DATA_DIR / "val.parquet")
    test_df  = pd.read_parquet(DATA_DIR / "test.parquet")

    print("Normalizing...")
    train_s, params = normalize_data(train_df, method="standard")
    val_s, _  = normalize_data(val_df, method="standard", params=params)
    test_s, _ = normalize_data(test_df, method="standard", params=params)

    all_results = {
        "with_features":    run_condition(train_s, val_s, test_s, use_time_features=True),
        "without_features": run_condition(train_s, val_s, test_s, use_time_features=False),
    }

    # Save JSON
    out_json = RESULTS_DIR / "ablation_features.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMetrics saved → {out_json}")

    # Generate plots
    print("\nGenerating plots...")
    plot_target_ablation(all_results)
    plot_per_domain_ablation(all_results)
    plot_improvement_pct(all_results)

    print("\n✓ Ablation study complete!")


if __name__ == "__main__":
    main()
