import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import feature_columns, normalize_data, target_columns

TARGET_CODE = "ES"
SOURCE_CODES = ["BE", "DE", "FR", "GR", "IT", "NL", "PT"]
RIDGE_ALPHA = 1.0

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train zero-shot baselines on the long-format dataset")
    p.add_argument("--pred_len", type=int, default=24)
    p.add_argument("--include_temporal", dest="include_temporal", action="store_true")
    p.add_argument("--no-include-temporal", dest="include_temporal", action="store_false")
    p.add_argument("--include_weather", dest="include_weather", action="store_true")
    p.add_argument("--no-include-weather", dest="include_weather", action="store_false")
    p.add_argument("--include_country_id", dest="include_country_id", action="store_true")
    p.add_argument("--no-include-country-id", dest="include_country_id", action="store_false")
    p.set_defaults(
        include_temporal=True,
        include_weather=True,
        include_country_id=True,
    )
    return p.parse_args()


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


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


def to_xy(df: pd.DataFrame, feature_cols: list[str], y_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    Y = df[y_cols].to_numpy(dtype=np.float32, copy=True)
    return X, Y


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
    }


def predict_daily_naive(df: pd.DataFrame, pred_len: int) -> np.ndarray:
    preds = []
    for horizon in range(1, pred_len + 1):
        source_lag = 24 - horizon
        if source_lag == 0:
            preds.append(df["demand"].to_numpy(dtype=np.float32))
        else:
            preds.append(df[f"lag_{source_lag}"].to_numpy(dtype=np.float32))
    return np.stack(preds, axis=1)


def evaluate_per_domain(
    model_name: str,
    predict_fn,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    y_cols: list[str],
) -> dict[str, dict[str, float]]:
    print(f"\n--- Model: {model_name} ---")
    results: dict[str, dict[str, float]] = {}
    for code in SOURCE_CODES + [TARGET_CODE]:
        domain_df = test_df[test_df["country_code"] == code].reset_index(drop=True)
        X_test, y_test = to_xy(domain_df, feature_cols, y_cols)
        y_pred = predict_fn(domain_df, X_test)
        results[DOMAIN_RESULT_KEYS[code]] = eval_metrics(y_test, y_pred)
    return results


def main() -> None:
    args = parse_args()

    data_dir = ROOT / "data" / "processed_long"
    models_dir = ROOT / "saved_models"
    results_dir = ROOT / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading long-format processed splits...")
    train_df = load_split(data_dir / "train.parquet")
    val_df = load_split(data_dir / "val.parquet")
    test_df = load_split(data_dir / "test.parquet")

    train_df = train_df[train_df["role"] == "source"].reset_index(drop=True)
    val_df = val_df[val_df["role"] == "source"].reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    y_cols = target_columns(args.pred_len)
    x_cols = feature_columns(
        train_df,
        include_temporal=args.include_temporal,
        include_weather=args.include_weather,
        include_country_id=args.include_country_id,
    )
    print(f"Feature count: {len(x_cols)}")

    train_scaled, val_scaled, test_scaled = scale_frames(train_df, val_df, test_df, x_cols, y_cols)
    X_train_src, y_train_src = to_xy(train_scaled, x_cols, y_cols)
    X_val_src, y_val_src = to_xy(val_scaled, x_cols, y_cols)

    all_results = {}

    all_results["Daily Naive"] = evaluate_per_domain(
        "Daily Naive",
        lambda domain_df, _: predict_daily_naive(domain_df, args.pred_len),
        test_scaled,
        x_cols,
        y_cols,
    )

    print(f"\nTraining Ridge Regression on source domains (alpha={RIDGE_ALPHA})...")
    ridge = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(X_train_src, y_train_src)
    all_results["Ridge Regression"] = evaluate_per_domain(
        "Ridge Regression",
        lambda _df, X: ridge.predict(X),
        test_scaled,
        x_cols,
        y_cols,
    )
    joblib.dump(ridge, models_dir / "baseline_ridge.joblib")
    print(f"Saved Ridge model -> {models_dir / 'baseline_ridge.joblib'}")

    print("\nTraining XGBoost on source domains...")
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
        X_train_src,
        y_train_src,
        eval_set=[(X_train_src, y_train_src), (X_val_src, y_val_src)],
        verbose=10,
    )
    all_results["XGBoost"] = evaluate_per_domain(
        "XGBoost",
        lambda _df, X: xgb.predict(X),
        test_scaled,
        x_cols,
        y_cols,
    )
    xgb.save_model(str(models_dir / "baseline_xgb.json"))
    print(f"Saved XGBoost model -> {models_dir / 'baseline_xgb.json'}")
    meta_path = models_dir / "baseline_xgb_features.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "feature_cols": x_cols,
                "target_cols": y_cols,
                "include_temporal": args.include_temporal,
                "include_weather": args.include_weather,
                "include_country_id": args.include_country_id,
            },
            f,
            indent=2,
        )
    print(f"Saved XGBoost metadata -> {meta_path}")

    results_path = results_dir / "baseline_metrics.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll metrics saved -> {results_path}")


if __name__ == "__main__":
    main()
