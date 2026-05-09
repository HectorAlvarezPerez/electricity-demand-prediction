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
