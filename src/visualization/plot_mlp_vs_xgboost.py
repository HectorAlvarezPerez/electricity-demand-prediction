"""
Generate a comparison figure between the latest persisted MLP run and the saved
XGBoost baseline, using real experiment artifacts instead of hardcoded values.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import normalize_data

MLFLOW_DB = ROOT / "mlflow.db"
BASELINE_RESULTS = ROOT / "results" / "baseline_metrics.json"
XGB_MODEL = ROOT / "saved_models" / "baseline_xgb.json"
XGB_META = ROOT / "saved_models" / "baseline_xgb_features.json"
OUT_JSON = ROOT / "results" / "mlp_xgboost_comparison.json"
OUT_FIG = ROOT / "docs" / "figures" / "mlp_vs_xgboost.png"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    diff = y_true - y_pred
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
    }


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_latest_mlp_metrics(experiment_name: str = "mlp_tabular_long") -> dict:
    required_keys = [
        "source_test_mae_norm",
        "source_test_rmse_norm",
        "target_test_mae_norm",
        "target_test_rmse_norm",
        "source_test_mae_mw",
        "source_test_rmse_mw",
        "target_test_mae_mw",
        "target_test_rmse_mw",
    ]
    with sqlite3.connect(MLFLOW_DB) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT m.run_uuid
            FROM metrics m
            JOIN runs r ON m.run_uuid = r.run_uuid
            JOIN experiments e ON r.experiment_id = e.experiment_id
            WHERE e.name = ? AND m.key = 'target_test_mae_norm'
            GROUP BY m.run_uuid
            ORDER BY MAX(m.timestamp) DESC
            LIMIT 1
            """,
            (experiment_name,),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"No MLflow run found for experiment '{experiment_name}'.")
        run_id = row[0]

        placeholders = ",".join("?" for _ in required_keys)
        cur.execute(
            f"""
            SELECT key, value
            FROM metrics
            WHERE run_uuid = ? AND key IN ({placeholders})
            """,
            [run_id, *required_keys],
        )
        values = {key: float(value) for key, value in cur.fetchall()}

        cur.execute(
            """
            SELECT m.step,
                   MAX(CASE WHEN m.key='source_val_mae' THEN m.value END) AS source_val_mae,
                   MAX(CASE WHEN m.key='target_val_mae' THEN m.value END) AS target_val_mae
            FROM metrics m
            WHERE m.run_uuid = ? AND m.key IN ('source_val_mae', 'target_val_mae')
            GROUP BY m.step
            ORDER BY m.step
            """,
            (run_id,),
        )
        history = [
            {
                "epoch": int(step),
                "source_val_mae": float(source_val_mae),
                "target_val_mae": float(target_val_mae),
            }
            for step, source_val_mae, target_val_mae in cur.fetchall()
        ]

    return {
        "run_id": run_id,
        "metrics": values,
        "history": history,
    }


def load_xgboost_metrics() -> dict:
    with open(XGB_META) as f:
        meta = json.load(f)

    data_dir = ROOT / "data" / "processed_long"
    train_df = load_split(data_dir / "train.parquet")
    test_df = load_split(data_dir / "test.parquet")

    train_df = train_df[train_df["role"] == "source"].reset_index(drop=True)
    source_test_df = test_df[test_df["role"] == "source"].reset_index(drop=True)
    target_test_df = test_df[test_df["role"] == "target"].reset_index(drop=True)

    x_cols = meta["feature_cols"]
    y_cols = meta["target_cols"]

    train_features, feature_params = normalize_data(train_df[x_cols], method="standard")
    train_targets, target_params = normalize_data(train_df[y_cols], method="standard")
    _ = train_features, train_targets

    source_test = source_test_df.copy()
    target_test = target_test_df.copy()
    source_test_features, _ = normalize_data(source_test[x_cols], method="standard", params=feature_params)
    target_test_features, _ = normalize_data(target_test[x_cols], method="standard", params=feature_params)
    source_test_targets, _ = normalize_data(source_test[y_cols], method="standard", params=target_params)
    target_test_targets, _ = normalize_data(target_test[y_cols], method="standard", params=target_params)

    source_test[x_cols] = source_test_features
    target_test[x_cols] = target_test_features
    source_test[y_cols] = source_test_targets
    target_test[y_cols] = target_test_targets

    model = XGBRegressor()
    model.load_model(str(XGB_MODEL))

    source_pred = model.predict(source_test[x_cols].to_numpy(dtype=np.float32, copy=True))
    target_pred = model.predict(target_test[x_cols].to_numpy(dtype=np.float32, copy=True))

    source_true = source_test[y_cols].to_numpy(dtype=np.float32, copy=True)
    target_true = target_test[y_cols].to_numpy(dtype=np.float32, copy=True)

    with open(BASELINE_RESULTS) as f:
        per_domain = json.load(f)["XGBoost"]

    return {
        "source_aggregate": compute_metrics(source_true, source_pred),
        "target_aggregate": compute_metrics(target_true, target_pred),
        "target_es": per_domain["target_es"],
    }


def plot_comparison(mlp: dict, xgb: dict) -> None:
    domains = ["Source", "Target (ES)"]
    model_order = ["XGBoost", "MLP"]
    mae = {
        "XGBoost": [xgb["source_aggregate"]["mae"], xgb["target_aggregate"]["mae"]],
        "MLP": [mlp["metrics"]["source_test_mae_norm"], mlp["metrics"]["target_test_mae_norm"]],
    }
    rmse = {
        "XGBoost": [xgb["source_aggregate"]["rmse"], xgb["target_aggregate"]["rmse"]],
        "MLP": [mlp["metrics"]["source_test_rmse_norm"], mlp["metrics"]["target_test_rmse_norm"]],
    }

    x = np.arange(len(domains))
    width = 0.34
    colors = {"XGBoost": "#1f77b4", "MLP": "#d62728"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    for ax, metric_name, values in zip(axes, ["MAE normalitzat", "RMSE normalitzat"], [mae, rmse]):
        for idx, model_name in enumerate(model_order):
            offset = (idx - 0.5) * width
            bars = ax.bar(
                x + offset,
                values[model_name],
                width=width,
                label=model_name,
                color=colors[model_name],
                alpha=0.9,
            )
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.004,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax.set_xticks(x, domains)
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Comparativa agregada entre XGBoost i MLP", fontsize=13)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    mlp = load_latest_mlp_metrics()
    xgb = load_xgboost_metrics()

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(
            {
                "mlp": mlp,
                "xgboost": xgb,
            },
            f,
            indent=2,
        )

    plot_comparison(mlp, xgb)
    print(f"Saved comparison metrics -> {OUT_JSON}")
    print(f"Saved figure -> {OUT_FIG}")


if __name__ == "__main__":
    main()
