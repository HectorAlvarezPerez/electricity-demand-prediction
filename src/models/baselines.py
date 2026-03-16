import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from src.data.dataset import ElectricityDemandDataset
from src.data.preprocess import normalize_data

def prepare_tabular_data(df, target_col, seq_len=168, pred_len=24):
    """
    Creates (X, Y) matrices directly from the dataframe.
    X is flattened (num_samples, seq_len * num_features).
    Y is (num_samples, pred_len)
    """
    dataset = ElectricityDemandDataset(df, seq_len=seq_len, pred_len=pred_len, target_col=target_col)
    
    # We can iterate through the dataset to collect pairs, but faster to do array slicing
    num_samples = len(dataset)
    num_features = df.shape[1]
    
    X = np.empty((num_samples, seq_len * num_features), dtype=np.float32)
    Y = np.empty((num_samples, pred_len), dtype=np.float32)
    
    for i in range(num_samples):
        x, y = dataset[i]
        X[i] = x.numpy().flatten()
        Y[i] = y.numpy().flatten()
        
    return X, Y

def main():
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data" / "processed"
    
    print("Loading datasets...")
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    
    print("Normalizing data...")
    # Normalize features (including targets for consistent input)
    train_scaled, scaler_params = normalize_data(train_df, method="standard")
    val_scaled, _ = normalize_data(val_df, method="standard", params=scaler_params)
    test_scaled, _ = normalize_data(test_df, method="standard", params=scaler_params)
    
    target_col = "target_es"
    seq_len = 168  # 1 week of history
    pred_len = 24  # Predict next 24 hours
    
    print(f"Target: {target_col}")
    print(f"Sequence Length: {seq_len}h, Prediction Length: {pred_len}h")
    
    print("Preparing Tabular Data...")
    X_train, Y_train = prepare_tabular_data(train_scaled, target_col, seq_len, pred_len)
    X_val, Y_val = prepare_tabular_data(val_scaled, target_col, seq_len, pred_len)
    X_test, Y_test = prepare_tabular_data(test_scaled, target_col, seq_len, pred_len)
    
    print(f"Train shapes - X: {X_train.shape}, Y: {Y_train.shape}")
    print(f"Test shapes -  X: {X_test.shape},  Y: {Y_test.shape}")
    
    # Baseline 1: Sequential Naive (Repeat the last 24h of target as prediction)
    print("\n--- Model: Seasonal/Daily Naive ---")
    target_idx = train_df.columns.get_loc(target_col)
    
    # The last 24 values of target in the sequence of 168h
    # In each un-flattened sample of shape (168, num_features), the target is at column target_idx.
    # We want the values from hour -24 to 0. 
    # Since X is flattened (seq_len * num_features), we can extract it by reshaping:
    X_test_reshaped = X_test.reshape(-1, seq_len, train_df.shape[1])
    # The last 24h
    Y_pred_naive_scaled = X_test_reshaped[:, -24:, target_idx]
    
    def evaluate(Y_true, Y_pred, model_name):
        rmse = root_mean_squared_error(Y_true, Y_pred)
        mae = mean_absolute_error(Y_true, Y_pred)
        print(f"[{model_name}] Standardized Space -> RMSE: {rmse:.4f} | MAE: {mae:.4f}")
        return rmse, mae

    evaluate(Y_test, Y_pred_naive_scaled, "Daily Naive")

    # Baseline 2: Linear Regression
    print("\n--- Model: Linear Regression ---")
    lr = LinearRegression(n_jobs=-1)
    lr.fit(X_train, Y_train)
    Y_pred_lr = lr.predict(X_test)
    evaluate(Y_test, Y_pred_lr, "Linear Regression")

    # Baseline 3: XGBoost (Lightweight for quick baseline)
    print("\n--- Model: XGBoost Regressor ---")
    # Using MultiOutputRegressor behavior native to XGBoost 
    # n_estimators=50 to make it quick, max_depth=5 is standard. tree_method="hist" is very fast.
    xgb = XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, tree_method="hist", n_jobs=-1)
    
    # Since XGBoost is heavy, we may subsample train set for quick baseline or train on full.
    print("(Training XGBoost might take a couple minutes on full data...)")
    xgb.fit(X_train, Y_train)
    Y_pred_xgb = xgb.predict(X_test)
    evaluate(Y_test, Y_pred_xgb, "XGBoost")

if __name__ == "__main__":
    main()
