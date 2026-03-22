"""
Few-shot fine-tuning for the XGBoost tabular forecasting baseline.

Workflow:
  1. Load a source-pretrained XGBoost checkpoint from src/models/baselines.py
  2. Select a small fraction of target-domain training samples
  3. Continue boosting on target only, validating on target_val
  4. Evaluate on target_test and compare against zero-shot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import normalize_data
from src.paths import METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, ensure_artifact_dirs

DEFAULT_XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "n_jobs": -1,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune a pretrained XGBoost model on the target domain")
    p.add_argument(
        "--pretrained_model",
        type=Path,
        default=MODELS_DIR / "baseline_xgb.json",
        help="Checkpoint produced by src/models/baselines.py",
    )
    p.add_argument(
        "--metadata_path",
        type=Path,
        default=MODELS_DIR / "baseline_xgb_features.json",
        help="Metadata JSON produced by src/models/baselines.py",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target_fraction", type=float, default=0.05)
    p.add_argument(
        "--sampling",
        choices=["head", "random"],
        default="head",
        help="How to pick the few-shot target subset",
    )
    p.add_argument(
        "--additional_estimators",
        type=int,
        default=50,
        help="Extra boosting rounds appended on top of the pretrained model",
    )
    p.add_argument("--learning_rate", type=float, default=0.02)
    p.add_argument("--max_depth", type=int, default=4)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample_bytree", type=float, default=0.8)
    p.add_argument("--eval_metric", type=str, default="rmse")
    p.add_argument("--verbose", type=int, default=10)
    return p.parse_args()


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_metadata(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def sample_target_train(df: pd.DataFrame, fraction: float, sampling: str, seed: int) -> pd.DataFrame:
    if not 0 < fraction <= 1:
        raise ValueError("--target_fraction must be in (0, 1].")

    df = df.sort_values("utc_timestamp").reset_index(drop=True)
    n_samples = max(1, int(np.ceil(len(df) * fraction)))

    if sampling == "head":
        sampled = df.iloc[:n_samples].copy()
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(df), size=n_samples, replace=False))
        sampled = df.iloc[idx].copy()

    return sampled.reset_index(drop=True)


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
    }


def to_xy(df: pd.DataFrame, feature_cols: list[str], target_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    Y = df[target_cols].to_numpy(dtype=np.float32, copy=True)
    return X, Y


def denormalize_targets(values: np.ndarray, target_params: dict) -> np.ndarray:
    mean = target_params["mean"].to_numpy(dtype=np.float32)
    std = target_params["std"].replace(0, 1).to_numpy(dtype=np.float32)
    return values * std + mean


def build_target_splits(
    root: Path,
    feature_cols: list[str],
    target_cols: list[str],
    *,
    target_fraction: float,
    sampling: str,
    seed: int,
):
    train_all = load_split(PROCESSED_DATA_DIR / "train.parquet")
    val_all = load_split(PROCESSED_DATA_DIR / "val.parquet")
    test_all = load_split(PROCESSED_DATA_DIR / "test.parquet")

    source_train = train_all[train_all["role"] == "source"].reset_index(drop=True)
    target_train_full = train_all[train_all["role"] == "target"].reset_index(drop=True)
    target_val = val_all[val_all["role"] == "target"].reset_index(drop=True)
    target_test = test_all[test_all["role"] == "target"].reset_index(drop=True)
    target_train = sample_target_train(target_train_full, target_fraction, sampling, seed)

    _, feature_params = normalize_data(source_train[feature_cols], method="standard")
    _, target_params = normalize_data(source_train[target_cols], method="standard")

    def apply_scaling(df: pd.DataFrame) -> pd.DataFrame:
        scaled = df.copy()
        scaled_features, _ = normalize_data(scaled[feature_cols], method="standard", params=feature_params)
        scaled_targets, _ = normalize_data(scaled[target_cols], method="standard", params=target_params)
        scaled[feature_cols] = scaled_features
        scaled[target_cols] = scaled_targets
        return scaled

    return (
        apply_scaling(target_train),
        apply_scaling(target_val),
        apply_scaling(target_test),
        target_params,
        len(target_train),
        len(target_train_full),
    )


def build_finetune_model(args: argparse.Namespace) -> XGBRegressor:
    params = {
        **DEFAULT_XGB_PARAMS,
        "n_estimators": args.additional_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "eval_metric": args.eval_metric,
    }
    return XGBRegressor(**params)


def main() -> None:
    args = parse_args()
    if not args.pretrained_model.exists():
        raise FileNotFoundError(args.pretrained_model)
    if not args.metadata_path.exists():
        raise FileNotFoundError(args.metadata_path)

    meta = load_metadata(args.metadata_path)
    feature_cols = meta["feature_cols"]
    target_cols = meta["target_cols"]

    print(f"Loading pretrained XGBoost checkpoint: {args.pretrained_model}")
    pretrained = XGBRegressor()
    pretrained.load_model(str(args.pretrained_model))

    (
        target_train,
        target_val,
        target_test,
        target_params,
        n_target_samples,
        n_target_full,
    ) = build_target_splits(
        ROOT,
        feature_cols,
        target_cols,
        target_fraction=args.target_fraction,
        sampling=args.sampling,
        seed=args.seed,
    )
    print(f"Target few-shot subset: {n_target_samples}/{n_target_full} samples ({args.target_fraction:.1%})")

    X_train, y_train = to_xy(target_train, feature_cols, target_cols)
    X_val, y_val = to_xy(target_val, feature_cols, target_cols)
    X_test, y_test = to_xy(target_test, feature_cols, target_cols)

    zero_val_preds = pretrained.predict(X_val)
    zero_test_preds = pretrained.predict(X_test)
    zero_val_metrics = eval_metrics(y_val, zero_val_preds)
    zero_test_metrics = eval_metrics(y_test, zero_test_preds)
    zero_test_preds_mw = denormalize_targets(zero_test_preds, target_params)
    zero_test_targets_mw = denormalize_targets(y_test, target_params)
    zero_test_metrics_mw = eval_metrics(zero_test_targets_mw, zero_test_preds_mw)

    print("\n--- Zero-shot target performance before fine-tuning ---")
    print(f"  Val MAE:  {zero_val_metrics['mae']:.4f}")
    print(f"  Test MAE: {zero_test_metrics['mae']:.4f}")
    print(f"  Test RMSE:{zero_test_metrics['rmse']:.4f}")
    print(f"  Test MAE: {zero_test_metrics_mw['mae']:.1f} MW")
    print(f"  Test RMSE:{zero_test_metrics_mw['rmse']:.1f} MW")

    finetuned = build_finetune_model(args)
    finetuned.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=args.verbose,
        xgb_model=str(args.pretrained_model),
    )

    target_test_preds = finetuned.predict(X_test)
    target_test_metrics_norm = eval_metrics(y_test, target_test_preds)
    target_test_preds_mw = denormalize_targets(target_test_preds, target_params)
    target_test_metrics_mw = eval_metrics(zero_test_targets_mw, target_test_preds_mw)
    improvement = {
        "mae_norm_abs": zero_test_metrics["mae"] - target_test_metrics_norm["mae"],
        "rmse_norm_abs": zero_test_metrics["rmse"] - target_test_metrics_norm["rmse"],
        "mae_mw_abs": zero_test_metrics_mw["mae"] - target_test_metrics_mw["mae"],
        "rmse_mw_abs": zero_test_metrics_mw["rmse"] - target_test_metrics_mw["rmse"],
    }
    improvement["mae_norm_pct"] = (
        improvement["mae_norm_abs"] / zero_test_metrics["mae"] * 100 if zero_test_metrics["mae"] else float("nan")
    )
    improvement["rmse_norm_pct"] = (
        improvement["rmse_norm_abs"] / zero_test_metrics["rmse"] * 100 if zero_test_metrics["rmse"] else float("nan")
    )
    improvement["mae_mw_pct"] = (
        improvement["mae_mw_abs"] / zero_test_metrics_mw["mae"] * 100 if zero_test_metrics_mw["mae"] else float("nan")
    )
    improvement["rmse_mw_pct"] = (
        improvement["rmse_mw_abs"] / zero_test_metrics_mw["rmse"] * 100 if zero_test_metrics_mw["rmse"] else float("nan")
    )

    print("\n--- Fine-tuned target test results (normalised) ---")
    print(f"  MAE:  {target_test_metrics_norm['mae']:.4f}")
    print(f"  RMSE: {target_test_metrics_norm['rmse']:.4f}")

    print("\n--- Fine-tuned target test results (MW) ---")
    print(f"  MAE:  {target_test_metrics_mw['mae']:.1f} MW")
    print(f"  RMSE: {target_test_metrics_mw['rmse']:.1f} MW")

    print("\n--- Improvement vs zero-shot target test ---")
    print(f"  Delta MAE (norm): {improvement['mae_norm_abs']:+.4f} ({improvement['mae_norm_pct']:+.2f}%)")
    print(f"  Delta RMSE (norm): {improvement['rmse_norm_abs']:+.4f} ({improvement['rmse_norm_pct']:+.2f}%)")
    print(f"  Delta MAE (MW):   {improvement['mae_mw_abs']:+.1f} MW ({improvement['mae_mw_pct']:+.2f}%)")
    print(f"  Delta RMSE (MW):  {improvement['rmse_mw_abs']:+.1f} MW ({improvement['rmse_mw_pct']:+.2f}%)")

    suffix = f"seed{args.seed}_frac{args.target_fraction:.3f}".replace(".", "p")
    ensure_artifact_dirs()

    model_path = MODELS_DIR / f"xgb_finetune_{suffix}.json"
    finetuned.save_model(str(model_path))

    evals_result = finetuned.evals_result()
    evals_path = METRICS_DIR / f"xgb_finetune_{suffix}_evals.json"
    with open(evals_path, "w") as f:
        json.dump(evals_result, f, indent=2)

    metrics_path = METRICS_DIR / f"xgb_finetune_{suffix}.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "pretrained_model": str(args.pretrained_model),
                "metadata_path": str(args.metadata_path),
                "target_fraction": args.target_fraction,
                "sampling": args.sampling,
                "target_samples": n_target_samples,
                "target_samples_full": n_target_full,
                "additional_estimators": args.additional_estimators,
                "learning_rate": args.learning_rate,
                "max_depth": args.max_depth,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
                "zero_shot_target_val": {
                    "metrics_norm": zero_val_metrics,
                },
                "zero_shot_target_test": {
                    "metrics_norm": zero_test_metrics,
                    "metrics_mw": zero_test_metrics_mw,
                },
                "finetuned_target_test": {
                    "metrics_norm": target_test_metrics_norm,
                    "metrics_mw": target_test_metrics_mw,
                },
                "improvement_vs_zero_shot": improvement,
                "model_path": str(model_path),
                "evals_path": str(evals_path),
            },
            f,
            indent=2,
        )

    joblib.dump(
        {
            "feature_cols": feature_cols,
            "target_cols": target_cols,
            "target_fraction": args.target_fraction,
            "sampling": args.sampling,
            "additional_estimators": args.additional_estimators,
        },
        MODELS_DIR / f"xgb_finetune_{suffix}_meta.joblib",
    )

    print(f"\nFine-tuned XGBoost model saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Eval history saved to {evals_path}")


if __name__ == "__main__":
    main()
