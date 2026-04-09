"""Profile XGBoost training, inference latency and memory usage."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmarking.common import (
    ModelBenchmarkOutput,
    compute_metrics,
    current_rss_mb,
    model_size_mb,
    parameter_count,
    profile_numpy_batches,
    save_json,
    cuda_peak_mb,
)
from src.data.preprocess import feature_columns, normalize_data, target_columns
from src.paths import METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, ensure_artifact_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile the XGBoost benchmark")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pred_len", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--warmup_batches", type=int, default=3)
    p.add_argument("--timed_batches", type=int, default=20)
    p.add_argument("--n_estimators", type=int, default=100)
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample_bytree", type=float, default=0.8)
    p.add_argument("--n_jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--output", type=Path, default=METRICS_DIR / "resource_benchmark" / "xgb_seed42.json")
    return p.parse_args()


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def scale_frames(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    y_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
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
    return train, val, test, feature_params, target_params


def to_xy(df: pd.DataFrame, feature_cols: list[str], y_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return (
        df[feature_cols].to_numpy(dtype=np.float32, copy=True),
        df[y_cols].to_numpy(dtype=np.float32, copy=True),
    )


def main() -> None:
    args = parse_args()
    ensure_artifact_dirs()

    print(f"[XGBoost] Starting profile | seed={args.seed} | n_estimators={args.n_estimators} | n_jobs={args.n_jobs}", flush=True)

    train_df = load_split(PROCESSED_DATA_DIR / "train.parquet")
    val_df = load_split(PROCESSED_DATA_DIR / "val.parquet")
    test_df = load_split(PROCESSED_DATA_DIR / "test.parquet")

    train_df = train_df[train_df["role"] == "source"].reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    y_cols = target_columns(args.pred_len)
    x_cols = feature_columns(
        train_df,
        include_temporal=True,
        include_weather=True,
        include_country_id=False,
    )

    train_scaled, val_scaled, test_scaled, feature_params, target_params = scale_frames(
        train_df, val_df, test_df, x_cols, y_cols
    )

    source_train = train_scaled[train_scaled["role"] == "source"].reset_index(drop=True)
    source_val = val_scaled[val_scaled["role"] == "source"].reset_index(drop=True)
    source_test = test_scaled[test_scaled["role"] == "source"].reset_index(drop=True)
    target_val = val_scaled[val_scaled["role"] == "target"].reset_index(drop=True)
    target_test = test_scaled[test_scaled["role"] == "target"].reset_index(drop=True)

    X_train, y_train = to_xy(source_train, x_cols, y_cols)
    X_source_val, y_source_val = to_xy(source_val, x_cols, y_cols)
    X_source_test, y_source_test = to_xy(source_test, x_cols, y_cols)
    X_target_val, y_target_val = to_xy(target_val, x_cols, y_cols)
    X_target_test, y_target_test = to_xy(target_test, x_cols, y_cols)

    model = XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        tree_method="hist",
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )

    print("[XGBoost] Training...", flush=True)
    train_t0 = time.perf_counter()
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_source_val, y_source_val)], verbose=False)
    train_time_s = time.perf_counter() - train_t0
    print(f"[XGBoost] Training done in {train_time_s:.1f}s", flush=True)

    predictions = {
        "source_train": model.predict(X_train),
        "source_val": model.predict(X_source_val),
        "source_test": model.predict(X_source_test),
        "target_val": model.predict(X_target_val),
        "target_test": model.predict(X_target_test),
    }

    metrics = {
        split_name: compute_metrics(y_true, y_pred)
        for split_name, (y_true, y_pred) in {
            "source_train": (y_train, predictions["source_train"]),
            "source_val": (y_source_val, predictions["source_val"]),
            "source_test": (y_source_test, predictions["source_test"]),
            "target_val": (y_target_val, predictions["target_val"]),
            "target_test": (y_target_test, predictions["target_test"]),
        }.items()
    }

    profile = {
        "source_test": profile_numpy_batches(
            X_source_test,
            model.predict,
            batch_size=args.batch_size,
            warmup_batches=args.warmup_batches,
            timed_batches=args.timed_batches,
        ),
        "target_test": profile_numpy_batches(
            X_target_test,
            model.predict,
            batch_size=args.batch_size,
            warmup_batches=args.warmup_batches,
            timed_batches=args.timed_batches,
        ),
    }
    print("[XGBoost] Inference profiling done", flush=True)

    model_path = MODELS_DIR / f"resource_xgb_seed{args.seed}.json"
    meta_path = MODELS_DIR / f"resource_xgb_seed{args.seed}_meta.json"
    model.save_model(str(model_path))
    save_json(
        meta_path,
        {
            "feature_cols": x_cols,
            "target_cols": y_cols,
            "feature_params": {k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in feature_params.items()},
            "target_params": {k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in target_params.items()},
            "include_country_id": False,
        },
    )

    output = ModelBenchmarkOutput(
        model_name="xgboost",
        seed=args.seed,
        feature_cols=x_cols,
        target_cols=y_cols,
        n_parameters=None,
        n_trainable_parameters=None,
        model_size_mb=model_size_mb(model_path),
        peak_rss_mb=current_rss_mb(),
        peak_vram_mb=cuda_peak_mb(),
        train_time_s=train_time_s,
        fit_metrics=metrics,
        inference=profile,
        artifact_paths={"model": str(model_path), "metadata": str(meta_path)},
    )

    save_json(args.output, output.to_dict())
    print(f"[XGBoost] Saved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
