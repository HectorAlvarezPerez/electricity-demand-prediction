"""Dataset loaders for the daily renewables benchmark."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphDataLoader

from src.data.dataset import create_dataloader
from src.data.preprocess import ALL_CODES, TARGET_CODE
from src.data.renewables import feature_columns, normalize_data, target_columns


def load_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(data_dir / "train.parquet"),
        pd.read_parquet(data_dir / "val.parquet"),
        pd.read_parquet(data_dir / "test.parquet"),
    )


def split_for_role(df: pd.DataFrame, role: str) -> pd.DataFrame:
    return df[df["role"] == role].reset_index(drop=True)


def scale_tabular_frames(
    train_source: pd.DataFrame,
    val_all: pd.DataFrame,
    test_all: pd.DataFrame,
    x_cols: list[str],
    y_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    train = train_source.copy()
    val = val_all.copy()
    test = test_all.copy()

    train_features, feature_params = normalize_data(train[x_cols])
    val_features, _ = normalize_data(val[x_cols], feature_params)
    test_features, _ = normalize_data(test[x_cols], feature_params)

    train_targets, target_params = normalize_data(train[y_cols])
    val_targets, _ = normalize_data(val[y_cols], target_params)
    test_targets, _ = normalize_data(test[y_cols], target_params)

    train[x_cols] = train_features
    val[x_cols] = val_features
    test[x_cols] = test_features
    train[y_cols] = train_targets
    val[y_cols] = val_targets
    test[y_cols] = test_targets
    return train, val, test, feature_params, target_params


def scale_all_frames_from_source(
    train_all: pd.DataFrame,
    val_all: pd.DataFrame,
    test_all: pd.DataFrame,
    x_cols: list[str],
    y_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    train = train_all.copy()
    val = val_all.copy()
    test = test_all.copy()
    train_source = train[train["role"] == "source"].reset_index(drop=True)

    _, feature_params = normalize_data(train_source[x_cols])
    _, target_params = normalize_data(train_source[y_cols])
    for frame in [train, val, test]:
        scaled_features, _ = normalize_data(frame[x_cols], feature_params)
        scaled_targets, _ = normalize_data(frame[y_cols], target_params)
        frame[x_cols] = scaled_features
        frame[y_cols] = scaled_targets
    return train, val, test, feature_params, target_params


def get_tabular_dataloaders(
    data_dir: Path,
    *,
    batch_size: int,
    include_external: bool,
    include_country_id: bool,
) -> tuple:
    train_all, val_all, test_all = load_splits(data_dir)
    train_source = split_for_role(train_all, "source")
    y_cols = target_columns()
    x_cols = feature_columns(
        train_source,
        include_temporal=True,
        include_external=include_external,
        include_country_id=include_country_id,
    )
    train_scaled, val_scaled, test_scaled, feature_params, target_params = scale_tabular_frames(
        train_source,
        val_all,
        test_all,
        x_cols,
        y_cols,
    )

    frames = {
        "source_train": train_scaled,
        "source_val": split_for_role(val_scaled, "source"),
        "source_test": split_for_role(test_scaled, "source"),
        "target_val": split_for_role(val_scaled, "target"),
        "target_test": split_for_role(test_scaled, "target"),
    }
    loaders = {
        name: create_dataloader(frame, x_cols, y_cols, batch_size=batch_size, shuffle=(name == "source_train"))
        for name, frame in frames.items()
    }
    return loaders, frames, x_cols, y_cols, feature_params, target_params


def build_static_graph(train_df: pd.DataFrame, nodes: list[str], top_k: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    pivot = train_df.pivot(index="date", columns="country_code", values="renewable_total_mwh").reindex(columns=nodes)
    corr = pivot.corr().fillna(0.0).to_numpy()
    edges: dict[tuple[int, int], float] = {}
    for i in range(len(nodes)):
        c = corr[i].copy()
        c[i] = -1.0
        for j in np.argsort(c)[-top_k:]:
            weight = float(corr[i, j])
            if weight > 0:
                edges[(i, int(j))] = weight
                edges[(int(j), i)] = weight
    if not edges:
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j:
                    edges[(i, j)] = 1e-4
    edge_index = torch.tensor(list(edges.keys()), dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(list(edges.values()), dtype=torch.float32)
    return edge_index, edge_weight


def extract_graphs(
    df: pd.DataFrame,
    x_cols: list[str],
    y_cols: list[str],
    nodes: list[str],
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> list[Data]:
    df = df.copy()
    df["country_code"] = pd.Categorical(df["country_code"], categories=nodes, ordered=True)
    df = df.sort_values(["date", "country_code"])
    counts = df.groupby("date", observed=False).size()
    valid_dates = counts[counts == len(nodes)].index
    df = df[df["date"].isin(valid_dates)]
    n_dates = len(valid_dates)
    n_nodes = len(nodes)
    x = df[x_cols].to_numpy(dtype=np.float32).reshape(n_dates, n_nodes, len(x_cols))
    y = df[y_cols].to_numpy(dtype=np.float32).reshape(n_dates, n_nodes, len(y_cols))
    return [
        Data(
            x=torch.tensor(x[i], dtype=torch.float32),
            y=torch.tensor(y[i], dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_weight,
            source_mask=source_mask,
            target_mask=target_mask,
        )
        for i in range(n_dates)
    ]


def get_graph_dataloaders(
    data_dir: Path,
    *,
    batch_size: int,
    include_external: bool,
) -> tuple:
    train_all, val_all, test_all = load_splits(data_dir)
    nodes = sorted(list(set(ALL_CODES)))
    source_mask = torch.tensor([node != TARGET_CODE for node in nodes], dtype=torch.bool)
    target_mask = torch.tensor([node == TARGET_CODE for node in nodes], dtype=torch.bool)
    y_cols = target_columns()
    x_cols = feature_columns(
        train_all,
        include_temporal=True,
        include_external=include_external,
        include_country_id=False,
    )

    train_scaled, val_scaled, test_scaled, feature_params, target_params = scale_all_frames_from_source(
        train_all,
        val_all,
        test_all,
        x_cols,
        y_cols,
    )
    # Use the unscaled training split for graph topology, matching the demand pipeline style.
    edge_index, edge_weight = build_static_graph(train_all, nodes=nodes, top_k=3)

    train_graphs = extract_graphs(train_scaled, x_cols, y_cols, nodes, edge_index, edge_weight, source_mask, target_mask)
    val_graphs = extract_graphs(val_scaled, x_cols, y_cols, nodes, edge_index, edge_weight, source_mask, target_mask)
    test_graphs = extract_graphs(test_scaled, x_cols, y_cols, nodes, edge_index, edge_weight, source_mask, target_mask)
    loaders = {
        "source_train": GraphDataLoader(train_graphs, batch_size=batch_size, shuffle=True),
        "val": GraphDataLoader(val_graphs, batch_size=batch_size, shuffle=False),
        "test": GraphDataLoader(test_graphs, batch_size=batch_size, shuffle=False),
    }
    return loaders, x_cols, y_cols, feature_params, target_params
