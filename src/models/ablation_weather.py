"""
Ablation study: impact of adding temperature as an exogenous feature.

Compares model performance in 3 conditions:
  1. Demand + temporal features (current baseline)
  2. Demand + temporal features + hourly temperature
  3. Demand only (no extra features)

Saves metrics to results/ablation_weather.json
Saves plots to docs/figures/ablation_weather_*.png

Run:
    python src/models/ablation_weather.py
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

# ── Constants ────────────────────────────────────────────────────────────────────
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

CODE_TO_COUNTRY = {
    "target_es": "ES", "source_be": "BE", "source_de": "DE",
    "source_fr": "FR", "source_gr": "GR", "source_it": "IT",
    "source_nl": "NL", "source_pt": "PT",
}

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
WEATHER_DIR = ROOT / "data" / "raw" / "weather"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "docs" / "figures"

SEQ_LEN = 168
PRED_LEN = 24


# ── Weather integration ──────────────────────────────────────────────────────────

def load_weather_for_country(country_code: str) -> pd.Series:
    """Load hourly temperature for a country code (e.g. 'ES')."""
    path = WEATHER_DIR / f"weather_{country_code}.csv"
    df = pd.read_csv(path, parse_dates=["utc_timestamp"], index_col="utc_timestamp")
    return df["temperature_2m"].astype(np.float32)


def add_weather_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add temperature columns (temp_{xx}) for each country to the dataframe."""
    df = df.copy()
    for domain_col in [TARGET_CODE] + SOURCE_CODES:
        cc = CODE_TO_COUNTRY[domain_col]
        prefix = cc.lower()
        temp = load_weather_for_country(cc)
        # Align to df index
        temp = temp.reindex(df.index)
        # Interpolate small gaps
        temp = temp.interpolate(method="time", limit=3)
        df[f"{prefix}_temp"] = temp
    # Drop rows where any temperature is missing (edges)
    df = df.dropna(subset=[f"{cc.lower()}_temp" for cc in CODE_TO_COUNTRY.values()])
    return df


# ── Data helpers ─────────────────────────────────────────────────────────────────

def prepare_domain_data(df, country_col, feature_mode, seq_len=SEQ_LEN, pred_len=PRED_LEN):
    """Build (X, Y) matrices for one country.
    
    feature_mode:
      'temporal'   -> demand + sin/cos + is_weekend
      'weather'    -> demand + sin/cos + is_weekend + temperature
      'demand_only' -> demand only
    """
    prefix = country_col.split("_")[-1]

    if feature_mode == "demand_only":
        cols = [country_col]
    elif feature_mode == "temporal":
        time_cols = [f"{prefix}_{f}" for f in TIME_FEATURES_BASE]
        cols = [country_col] + time_cols
    elif feature_mode == "weather":
        time_cols = [f"{prefix}_{f}" for f in TIME_FEATURES_BASE]
        cols = [country_col] + time_cols + [f"{prefix}_temp"]
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    df_country = df[cols]
    dataset = ElectricityDemandDataset(df_country, seq_len=seq_len,
                                       pred_len=pred_len, target_col=country_col)
    n = len(dataset)
    X = np.empty((n, seq_len * len(cols)), dtype=np.float32)
    Y = np.empty((n, pred_len), dtype=np.float32)
    for i in tqdm(range(n), desc=f"  {country_col} ({feature_mode})", leave=False):
        x, y = dataset[i]
        X[i] = x.numpy().flatten()
        Y[i] = y.numpy().flatten()
    return X, Y


def build_multidomain(df, codes, feature_mode):
    Xs, Ys = [], []
    for c in codes:
        X, Y = prepare_domain_data(df, c, feature_mode)
        Xs.append(X); Ys.append(Y)
    return np.vstack(Xs), np.vstack(Ys)


def eval_metrics(Y_true, Y_pred):
    return {
        "mae": float(mean_absolute_error(Y_true, Y_pred)),
        "rmse": float(root_mean_squared_error(Y_true, Y_pred)),
    }


def evaluate_model(name, predict_fn, test_df, feature_mode):
    results = {}
    for code in tqdm(SOURCE_CODES, desc=f"  Eval {name}"):
        Xt, Yt = prepare_domain_data(test_df, code, feature_mode)
        results[code] = eval_metrics(Yt, predict_fn(Xt, code))
    Xt, Yt = prepare_domain_data(test_df, TARGET_CODE, feature_mode)
    results[TARGET_CODE] = eval_metrics(Yt, predict_fn(Xt, TARGET_CODE))
    return results


# ── Training ─────────────────────────────────────────────────────────────────────

def run_condition(train_df, val_df, test_df, feature_mode):
    tag = feature_mode.upper()
    print(f"\n{'='*60}")
    print(f"  CONDITION: {tag}")
    print(f"{'='*60}")

    print("Preparing training data...")
    X_train, Y_train = build_multidomain(train_df, SOURCE_CODES, feature_mode)
    print(f"  Train shapes — X: {X_train.shape}, Y: {Y_train.shape}")

    print("Preparing validation data...")
    X_val, Y_val = build_multidomain(val_df, SOURCE_CODES, feature_mode)

    results = {}

    # Daily Naive
    print(f"\n[{tag}] Evaluating Daily Naive...")
    n_feat = X_train.shape[1] // SEQ_LEN
    def predict_naive(X, _col):
        X_r = X.reshape(-1, SEQ_LEN, n_feat)
        return X_r[:, -PRED_LEN:, 0]
    results["Daily Naive"] = evaluate_model("Daily Naive", predict_naive,
                                            test_df, feature_mode)
    print(f"  Target MAE: {results['Daily Naive'][TARGET_CODE]['mae']:.4f}")

    # Linear Regression
    print(f"\n[{tag}] Training Linear Regression...")
    lr = LinearRegression(n_jobs=-1)
    lr.fit(X_train, Y_train)
    results["Linear Regression"] = evaluate_model(
        "Linear Regression", lambda X, _: lr.predict(X), test_df, feature_mode)
    print(f"  Target MAE: {results['Linear Regression'][TARGET_CODE]['mae']:.4f}")

    # XGBoost
    print(f"\n[{tag}] Training XGBoost...")
    xgb = XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.1,
                        tree_method="hist", n_jobs=-1)
    xgb.fit(X_train, Y_train,
            eval_set=[(X_train, Y_train), (X_val, Y_val)],
            verbose=10)
    results["XGBoost"] = evaluate_model(
        "XGBoost", lambda X, _: xgb.predict(X), test_df, feature_mode)
    print(f"  Target MAE: {results['XGBoost'][TARGET_CODE]['mae']:.4f}")

    return results


# ── Plotting ─────────────────────────────────────────────────────────────────────

CONDITION_LABELS = {
    "temporal": "Temporal",
    "weather": "Temporal + Temperatura",
}
CONDITION_COLORS = {
    "temporal": "#2c7fb8",
    "weather": "#41ae76",
}


def plot_target_weather(all_results):
    """Horizontal bars: temporal vs temporal+weather for LR and XGBoost."""
    models = ["XGBoost", "Linear Regression"]
    conditions = ["temporal", "weather"]

    y = np.arange(len(models))
    h = 0.3

    fig, ax = plt.subplots(figsize=(10, 4))

    for i, cond in enumerate(conditions):
        vals = [all_results[cond][m][TARGET_CODE]["mae"] for m in models]
        offset = (i - 0.5) * h
        bars = ax.barh(y + offset, vals, h, label=CONDITION_LABELS[cond],
                       color=CONDITION_COLORS[cond], edgecolor="white", linewidth=0.5)
        for bar in bars:
            ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                    f"{bar.get_width():.4f}", va="center", fontsize=11,
                    fontweight="bold", color=CONDITION_COLORS[cond])

    # Delta labels
    for j, m in enumerate(models):
        mae_t = all_results["temporal"][m][TARGET_CODE]["mae"]
        mae_w = all_results["weather"][m][TARGET_CODE]["mae"]
        delta = mae_w - mae_t
        sign = "+" if delta >= 0 else ""
        mid_x = max(mae_t, mae_w) + 0.04
        color = "#c0392b" if delta > 0 else "#27ae60"
        ax.annotate(f"Δ {sign}{delta:.3f}",
                    xy=(mid_x, y[j]),
                    fontsize=11, fontweight="bold", color=color,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="#f0fef0" if delta <= 0 else "#fef0ef",
                              edgecolor=color, linewidth=1.2))

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=13)
    ax.set_xlabel("MAE (Normalitzat) — Target Zero-Shot (ES)", fontsize=12)
    ax.set_title("Impacte d'afegir Temperatura com a Feature", fontsize=14)
    ax.legend(fontsize=11, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.15))
    ax.invert_yaxis()
    fig.tight_layout()

    out = FIGURES_DIR / "ablation_weather_target.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved → {out}")


def plot_per_domain_weather(all_results):
    """Per-domain MAE: temporal vs temporal+weather, side by side."""
    conditions = ["temporal", "weather"]
    domains = list(DOMAIN_LABELS.keys())
    labels = [DOMAIN_LABELS[d] for d in domains]
    models = ["Linear Regression", "XGBoost"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    for ax, model in zip(axes, models):
        x = np.arange(len(domains))
        w = 0.35
        for i, cond in enumerate(conditions):
            vals = [all_results[cond][model][d]["mae"] for d in domains]
            offset = (i - 0.5) * w
            ax.bar(x + offset, vals, w, label=CONDITION_LABELS[cond],
                   color=CONDITION_COLORS[cond])
        ax.set_title(model, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
        ax.set_ylabel("MAE (Normalitzat)", fontsize=11)
        ax.legend(fontsize=9)

    fig.suptitle("MAE per País — Temporal vs Temporal + Temperatura", fontsize=14, y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / "ablation_weather_per_domain.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved → {out}")


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # ── Condition 1: temporal only → load from existing results ──────────
    existing_path = RESULTS_DIR / "ablation_features.json"
    if existing_path.exists():
        print(f"Loading temporal-only results from {existing_path}...")
        with open(existing_path) as f:
            existing = json.load(f)
        all_results["temporal"] = existing["with_features"]
        print("  ✓ Loaded (no retraining needed)")
    else:
        raise FileNotFoundError(
            f"{existing_path} not found. Run ablation_features.py first.")

    # ── Condition 2: temporal + weather → train ─────────────────────────
    print("\nLoading datasets...")
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    val_df   = pd.read_parquet(DATA_DIR / "val.parquet")
    test_df  = pd.read_parquet(DATA_DIR / "test.parquet")

    print("Adding weather data...")
    train_w = add_weather_to_df(train_df)
    val_w   = add_weather_to_df(val_df)
    test_w  = add_weather_to_df(test_df)
    print(f"  Train: {train_df.shape} → {train_w.shape}")
    print(f"  Val:   {val_df.shape} → {val_w.shape}")
    print(f"  Test:  {test_df.shape} → {test_w.shape}")

    print("Normalizing (weather data)...")
    train_sw, params_w = normalize_data(train_w, method="standard")
    val_sw, _  = normalize_data(val_w, method="standard", params=params_w)
    test_sw, _ = normalize_data(test_w, method="standard", params=params_w)

    all_results["weather"] = run_condition(train_sw, val_sw, test_sw, "weather")

    # Save JSON
    out_json = RESULTS_DIR / "ablation_weather.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMetrics saved → {out_json}")

    # Generate plots
    print("\nGenerating plots...")
    plot_target_weather(all_results)
    plot_per_domain_weather(all_results)

    print("\n✓ Weather ablation complete!")


if __name__ == "__main__":
    main()
