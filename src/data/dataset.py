import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class TabularForecastDataset(Dataset):
    """Simple tabular dataset over explicit lag features and multi-step targets."""

    def __init__(self, df: pd.DataFrame, feature_cols: list[str], target_cols: list[str]):
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.features = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
        self.targets = df[target_cols].to_numpy(dtype=np.float32, copy=True)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.features[idx])
        y = torch.from_numpy(self.targets[idx])
        return x, y


class ElectricityDemandDataset(Dataset):
    """
    Legacy sliding-window dataset kept temporarily for compatibility.

    The main pipeline now uses TabularForecastDataset with explicit lag columns.
    """

    def __init__(self, data, seq_len=168, pred_len=24, target_col=None):
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
        x = self.values[idx : idx + self.seq_len]
        if self.target_idx is not None:
            y = self.values[
                idx + self.seq_len : idx + self.seq_len + self.pred_len,
                self.target_idx,
            ].reshape(-1)
        else:
            y = self.values[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def create_dataloader(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    *,
    batch_size: int = 64,
    shuffle: bool = False,
) -> DataLoader:
    dataset = TabularForecastDataset(df, feature_cols, target_cols)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
