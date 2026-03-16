import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from src.data.dataset import ElectricityDemandDataset
from src.data.preprocess import normalize_data

SOURCE_CODES = ["source_be", "source_de", "source_fr", "source_gr", "source_it", "source_nl", "source_pt"]
TARGET_CODE = "target_es"
TIME_FEATURES_BASE = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend"]

def prepare_domain_data(df, country_col, seq_len=168, pred_len=24):
    """
    Creates (X, Y) matrices for a specific country using its demand and shared time features.
    """
    # Select columns: country's demand + time features
    # the datasets have temporal features prefixed by the country code, e.g., 'es_hour_sin'
    prefix = country_col.split("_")[-1]
    time_cols = [f"{prefix}_{feat}" for feat in TIME_FEATURES_BASE]
    
    cols = [country_col] + time_cols
    df_country = df[cols]
    
    dataset = ElectricityDemandDataset(df_country, seq_len=seq_len, pred_len=pred_len, target_col=country_col)
    
    num_samples = len(dataset)
    num_features = len(cols)
    
    X = np.empty((num_samples, seq_len * num_features), dtype=np.float32)
    Y = np.empty((num_samples, pred_len), dtype=np.float32)
    
    # We flatten the input sequence so a classical model (LR/XGB) can digest it
    for i in tqdm(range(num_samples), desc=f"Flattening {country_col}", leave=False):
        x, y = dataset[i]
        X[i] = x.numpy().flatten()
        Y[i] = y.numpy().flatten()
        
    return X, Y

def build_multidomain_dataset(df, country_cols, seq_len=168, pred_len=24):
    """
    Stacks datasets from multiple source countries into a single large training set.
    """
    X_list, Y_list = [], []
    for col in country_cols:
        X, Y = prepare_domain_data(df, col, seq_len, pred_len)
        X_list.append(X)
        Y_list.append(Y)
        
    return np.vstack(X_list), np.vstack(Y_list)

def evaluate(Y_true, Y_pred, prefix):
    rmse = root_mean_squared_error(Y_true, Y_pred)
    mae = mean_absolute_error(Y_true, Y_pred)
    print(f"{prefix} MAE: {mae:.4f} | RMSE: {rmse:.4f} (Standardized)")
    return {"mae": float(mae), "rmse": float(rmse)}

def evaluate_model(model_name, model_func, test_df, seq_len, pred_len):
    """Evaluates a model on all source domains and the target domain (zero-shot).
    Returns a dict with all metrics keyed by domain code."""
    print(f"\n--- Model: {model_name} ---")
    results = {}
    
    # 1. Evaluate on each Source Domain
    for src_code in tqdm(SOURCE_CODES, desc=f"Evaluating {model_name} on Source Domains"):
        X_test_src, Y_test_src = prepare_domain_data(test_df, src_code, seq_len, pred_len)
        Y_pred_src = model_func(X_test_src, src_code)
        metrics = evaluate(Y_test_src, Y_pred_src, f"[{model_name}] Source ({src_code})")
        results[src_code] = metrics
        
    # 2. Evaluate on Target Domain (Zero-Shot)
    X_test_tgt, Y_test_tgt = prepare_domain_data(test_df, TARGET_CODE, seq_len, pred_len)
    Y_pred_tgt = model_func(X_test_tgt, TARGET_CODE)
    metrics = evaluate(Y_test_tgt, Y_pred_tgt, f"[{model_name}] TARGET ZERO-SHOT (ES)")
    results[TARGET_CODE] = metrics
    
    return results

def main():
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data" / "processed"
    models_dir = root / "saved_models"
    results_dir = root / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading datasets...")
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    
    print("Normalizing data...")
    # Normalizing column by column. This is effectively "Domain-Specific Normalization" 
    # since each country is normalized by its own mean and variance during the training period.
    train_scaled, scaler_params = normalize_data(train_df, method="standard")
    val_scaled, _ = normalize_data(val_df, method="standard", params=scaler_params)
    test_scaled, _ = normalize_data(test_df, method="standard", params=scaler_params)
    
    seq_len = 168  # 1 week of history
    pred_len = 24  # Predict next 24 hours
    
    print(f"Sequence Length: {seq_len}h, Prediction Length: {pred_len}h")
    
    # 1. Prepare Training Data using ONLY European sources
    print("\nPreparing Source Domain (Europe) Training Data...")
    X_train_src, Y_train_src = build_multidomain_dataset(train_scaled, SOURCE_CODES, seq_len, pred_len)
    print(f"Source Train shapes - X: {X_train_src.shape}, Y: {Y_train_src.shape}")
    
    print("\nPreparing Source Domain (Europe) Validation Data (for XGBoost tracking)...")
    X_val_src, Y_val_src = build_multidomain_dataset(val_scaled, SOURCE_CODES, seq_len, pred_len)
    
    # Dict to collect all results
    all_results = {}
    
    # Baseline 0: Seasonal/Daily Naive
    def predict_naive(X_test, country_col):
        # We reshape to (batch, seq_len, features). Feature 0 is the demand.
        X_test_reshaped = X_test.reshape(-1, seq_len, len([country_col] + TIME_FEATURES_BASE))
        return X_test_reshaped[:, -24:, 0]
        
    all_results["Daily Naive"] = evaluate_model("Daily Naive", predict_naive, test_scaled, seq_len, pred_len)

    # Baseline 1: Linear Regression
    print("\nTraining Linear Regression on combined Source domains...")
    lr = LinearRegression(n_jobs=-1)
    lr.fit(X_train_src, Y_train_src)
    all_results["Linear Regression"] = evaluate_model("Linear Regression", lambda X, _: lr.predict(X), test_scaled, seq_len, pred_len)
    
    # Save LR model
    joblib.dump(lr, models_dir / "baseline_lr.joblib")
    print(f"Linear Regression model saved to {models_dir / 'baseline_lr.joblib'}")

    # Baseline 2: XGBoost Regressor
    print("\nTraining XGBoost Regressor on combined Source domains...")
    # tree_method="hist" makes training on ~500k samples with 1300 features fast. 
    # n_estimators=50 is low for a quick baseline.
    xgb = XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, tree_method="hist", n_jobs=-1)
    print("(Training XGBoost: printing progress every 5 trees)")
    xgb.fit(
        X_train_src, Y_train_src,
        eval_set=[(X_train_src, Y_train_src), (X_val_src, Y_val_src)],
        verbose=5
    )
    all_results["XGBoost"] = evaluate_model("XGBoost", lambda X, _: xgb.predict(X), test_scaled, seq_len, pred_len)
    
    # Save XGBoost model
    xgb.save_model(str(models_dir / "baseline_xgb.json"))
    print(f"XGBoost model saved to {models_dir / 'baseline_xgb.json'}")
    
    # Save all metrics to JSON
    results_path = results_dir / "baseline_metrics.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll metrics saved to {results_path}")

if __name__ == "__main__":
    main()
