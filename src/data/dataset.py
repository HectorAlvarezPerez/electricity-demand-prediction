import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from src.data.preprocess import normalize_data, add_temporal_features

class ElectricityDemandDataset(Dataset):
    """
    Dataset for electricity demand forecasting.
    Creates sliding windows from the time series data.
    """
    def __init__(self, data, seq_len=168, pred_len=24, target_col=None):
        """
        Args:
            data (pd.DataFrame or np.ndarray): The time series data. Features should be in columns.
            seq_len (int): Length of the input sequence (e.g., 168 hours = 1 week).
            pred_len (int): Length of the prediction sequence (e.g., 24 hours = 1 day).
            target_col (str or int): The specific column to predict. If None, predicts all columns.
        """
        # Convert DataFrame to numpy array if necessary
        if isinstance(data, pd.DataFrame):
            self.values = data.values
            self.columns = data.columns
            if target_col is not None and isinstance(target_col, str):
                self.target_idx = data.columns.get_loc(target_col)
            else:
                self.target_idx = target_col
        else:
            self.values = data
            self.target_idx = target_col
            
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = len(self.values)
        
    def __len__(self):
        return self.total_len - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        # Input sequence (features and target history)
        x = self.values[idx : idx + self.seq_len]
        
        # Target sequence
        if self.target_idx is not None:
            y = self.values[idx + self.seq_len : idx + self.seq_len + self.pred_len, self.target_idx]
            y = y.reshape(-1) # [pred_len]
        else:
            y = self.values[idx + self.seq_len : idx + self.seq_len + self.pred_len]
            
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def create_dataloaders(df, seq_len=168, pred_len=24, target_col=None, batch_size=32, train_split=0.7, val_split=0.15, scale_data=True, time_features=True):
    """
    Splits the dataframe, optionally adds temporal features, scales the data avoiding leakage,
    and creates train, validation, and test dataloaders.
    """
    df_processed = df.copy()
    
    if time_features:
        df_processed = add_temporal_features(df_processed)
        
    n = len(df_processed)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))
    
    train_data = df_processed.iloc[:train_end].copy()
    val_data = df_processed.iloc[train_end:val_end].copy()
    test_data = df_processed.iloc[val_end:].copy()
    
    scaler_params = None
    if scale_data:
        train_data, scaler_params = normalize_data(train_data, method='standard')
        val_data, _ = normalize_data(val_data, method='standard', params=scaler_params)
        test_data, _ = normalize_data(test_data, method='standard', params=scaler_params)
    
    train_dataset = ElectricityDemandDataset(train_data, seq_len, pred_len, target_col)
    val_dataset = ElectricityDemandDataset(val_data, seq_len, pred_len, target_col)
    test_dataset = ElectricityDemandDataset(test_data, seq_len, pred_len, target_col)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler_params
