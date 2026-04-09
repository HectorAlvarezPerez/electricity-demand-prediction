"""
Graph dataset builder for the explicit-lag forecasting pipeline.
Constructs a static 8-node graph from temporal correlation, and reshapes
the long-format dataframe into a sequence of torch_geometric `Data` objects.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.data.preprocess import feature_columns, normalize_data, target_columns, ALL_CODES, TARGET_CODE
from src.paths import PROCESSED_DATA_DIR

def load_split(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def scale_frames_for_graph(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    train = train_df.copy()
    val = val_df.copy()
    test = test_df.copy()

    train_features, feature_params = normalize_data(train[feature_cols], method="standard")
    val_features, _ = normalize_data(val[feature_cols], method="standard", params=feature_params)
    test_features, _ = normalize_data(test[feature_cols], method="standard", params=feature_params)

    train_targets, target_params = normalize_data(train[target_cols], method="standard")
    val_targets, _ = normalize_data(val[target_cols], method="standard", params=target_params)
    test_targets, _ = normalize_data(test[target_cols], method="standard", params=target_params)

    train[feature_cols] = train_features
    val[feature_cols] = val_features
    test[feature_cols] = test_features
    train[target_cols] = train_targets
    val[target_cols] = val_targets
    test[target_cols] = test_targets
    return train, val, test, feature_params, target_params


def build_static_graph(train_df: pd.DataFrame, nodes: list[str], top_k: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build static graph using demand correlation in the training set.
    Connects each node to its top_k most correlated neighbors.
    Symmetrizes the graph and assigns the positive correlation as edge weight.
    """
    # Calculate correlation based on the original demand (before or after scaling is identical for Pearson corr)
    demand_df = train_df.pivot(index="utc_timestamp", columns="country_code", values="demand")[nodes]
    corr = demand_df.corr().values
    
    num_nodes = len(nodes)
    unique_edges = {}
    
    for i in range(num_nodes):
        c = corr[i].copy()
        c[i] = -1.0 # Ignore self-loop
        top_indices = np.argsort(c)[-top_k:]
        for j in top_indices:
            unique_edges[(i, j)] = corr[i, j]
            unique_edges[(j, i)] = corr[j, i] # ensure symmetry physically although corr is symmetric

    # Filter to only keep positive weights
    filtered_edges = []
    filtered_weights = []
    for (i, j), w in unique_edges.items():
        if w > 0:
            filtered_edges.append((i, j))
            filtered_weights.append(w)
            
    if not filtered_edges:
        # Fallback fully connected if something goes completely wrong
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    filtered_edges.append((i, j))
                    filtered_weights.append(1e-4)

    edge_index = torch.tensor(filtered_edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(filtered_weights, dtype=torch.float32)
    return edge_index, edge_weight


def extract_graphs(
    df: pd.DataFrame, 
    x_cols: list[str], 
    y_cols: list[str], 
    nodes: list[str], 
    edge_index: torch.Tensor, 
    edge_weight: torch.Tensor,
    source_mask: torch.Tensor,
    target_mask: torch.Tensor
) -> list[Data]:
    """
    Convert a dataframe of all nodes over time into a sequence of Data objects.
    """
    # Sort strictly by timestamp and node order
    df = df.copy()
    # Categorize country code based on provided nodes list to ensure consistent sorting
    df["country_code"] = pd.Categorical(df["country_code"], categories=nodes, ordered=True)
    df = df.sort_values(["utc_timestamp", "country_code"])
    
    # Filter to only timestamps that have exactly `len(nodes)` complete rows.
    counts = df.groupby("utc_timestamp", observed=False).size()
    valid_ts = counts[counts == len(nodes)].index
    df_valid = df[df["utc_timestamp"].isin(valid_ts)]
    
    num_ts = len(valid_ts)
    num_nodes = len(nodes)
    
    X = df_valid[x_cols].to_numpy().reshape(num_ts, num_nodes, len(x_cols))
    Y = df_valid[y_cols].to_numpy().reshape(num_ts, num_nodes, len(y_cols))
    
    graphs = []
    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)
    
    for i in range(num_ts):
        data = Data(
            x=X_t[i],
            y=Y_t[i],
            edge_index=edge_index,
            edge_attr=edge_weight,
            source_mask=source_mask,
            target_mask=target_mask
        )
        graphs.append(data)
    
    return graphs


def get_graph_dataloaders(
    root: Path,
    *,
    pred_len: int,
    batch_size: int,
    include_temporal: bool,
    include_weather: bool,
) -> tuple:
    # 1. Load data
    train_all = load_split(PROCESSED_DATA_DIR / "train.parquet")
    val_all = load_split(PROCESSED_DATA_DIR / "val.parquet")
    test_all = load_split(PROCESSED_DATA_DIR / "test.parquet")
    
    # 2. Determine node order explicitly
    nodes = sorted(list(set(ALL_CODES)))
    
    # 3. Create masks
    source_mask = torch.tensor([n != TARGET_CODE for n in nodes], dtype=torch.bool)
    target_mask = torch.tensor([n == TARGET_CODE for n in nodes], dtype=torch.bool)
    
    # 4. Feature and Target columns
    y_cols = target_columns(pred_len)
    
    # Extract the base feature columns from train_all
    # We explicitly do NOT include country_id encoding for the GNN
    all_x_cols = feature_columns(
        train_all,
        include_temporal=include_temporal,
        include_weather=include_weather,
        include_country_id=False,  # <--- Crucial difference for GNN
    )
    
    # 5. Build static graph using train_all
    edge_index, edge_weight = build_static_graph(train_all, nodes=nodes, top_k=3)
    
    # 6. Global scaling based on all domains together across train
    (
        train_scaled,
        val_scaled,
        test_scaled,
        feature_params,
        target_params,
    ) = scale_frames_for_graph(
        train_all,
        val_all,
        test_all,
        all_x_cols,
        y_cols,
    )
    
    # 7. Convert to lists of torch_geometric Data objects
    train_graphs = extract_graphs(train_scaled, all_x_cols, y_cols, nodes, edge_index, edge_weight, source_mask, target_mask)
    val_graphs = extract_graphs(val_scaled, all_x_cols, y_cols, nodes, edge_index, edge_weight, source_mask, target_mask)
    test_graphs = extract_graphs(test_scaled, all_x_cols, y_cols, nodes, edge_index, edge_weight, source_mask, target_mask)
    
    # 8. Create DataLoaders
    # Note: torch_geometric.loader.DataLoader correctly batches independent graphs 
    # to form a giant disconnected graph batch
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    
    # For compatibility we return train, and separately we can just use val/test 
    # filtering by mask during evaluation. 
    # We'll return just one val_loader and test_loader because the evaluation logic 
    # can calculate both source and target loss by simply selecting the right mask.
    return (
        train_loader,
        val_loader,
        test_loader,
        all_x_cols,
        y_cols,
        feature_params,
        target_params,
    )

