"""
Training loop per a models de predicció de demanda elèctrica.

Ús:
    python src/train.py                        # defaults (MLP, target_es, 168→24)
    python src/train.py --pred_len 48          # 48h ahead
    python src/train.py --epochs 50 --lr 5e-4  # custom
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import ElectricityDemandDataset
from src.data.preprocess import normalize_data
from src.models.mlp_baseline import MLPBaseline


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(
    preds: np.ndarray, targets: np.ndarray
) -> dict[str, float]:
    """MAE, RMSE, MAPE sobre arrays 1-D o 2-D flatten."""
    preds = preds.flatten()
    targets = targets.flatten()
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    # MAPE: evita div/0 amb mask
    mask = np.abs(targets) > 1e-3
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((preds[mask] - targets[mask]) / targets[mask])) * 100)
    else:
        mape = float("nan")
    return {"mae": mae, "rmse": rmse, "mape": mape}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df


def build_loaders(
    root: Path,
    seq_len: int,
    pred_len: int,
    target_col: str,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Carrega parquets, normalitza amb stats de train, crea DataLoaders."""
    train_df = load_split(root / "data" / "processed" / "train.parquet")
    val_df = load_split(root / "data" / "processed" / "val.parquet")
    test_df = load_split(root / "data" / "processed" / "test.parquet")

    # Normalitzar amb stats de train (evita data leakage)
    train_df, scaler_params = normalize_data(train_df, method="standard")
    val_df, _ = normalize_data(val_df, method="standard", params=scaler_params)
    test_df, _ = normalize_data(test_df, method="standard", params=scaler_params)

    train_ds = ElectricityDemandDataset(train_df, seq_len, pred_len, target_col)
    val_ds = ElectricityDemandDataset(val_df, seq_len, pred_len, target_col)
    test_ds = ElectricityDemandDataset(test_df, seq_len, pred_len, target_col)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler_params


# ---------------------------------------------------------------------------
# Train / evaluate one epoch
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds, all_targets = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item()
        n_batches += 1
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())
    avg_loss = total_loss / max(n_batches, 1)
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return avg_loss, preds, targets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train electricity demand forecasting model")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seq_len", type=int, default=168, help="Input window (hours)")
    p.add_argument("--pred_len", type=int, default=24, help="Forecast horizon (hours)")
    p.add_argument("--target_col", type=str, default="target_es")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 256, 128])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--experiment_name", type=str, default="mlp_baseline")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    print("Loading data...")
    train_loader, val_loader, test_loader, scaler_params = build_loaders(
        root, args.seq_len, args.pred_len, args.target_col, args.batch_size
    )

    # Model
    n_features = train_loader.dataset.values.shape[1]
    model = MLPBaseline(
        n_features=n_features,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # MLflow
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run():
        mlflow.log_params(vars(args))
        mlflow.log_param("n_features", n_features)
        mlflow.log_param("device", str(device))

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
            elapsed = time.time() - t0

            val_metrics = compute_metrics(val_preds, val_targets)

            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | "
                f"val_mae={val_metrics['mae']:.4f} | "
                f"val_rmse={val_metrics['rmse']:.4f} | "
                f"{elapsed:.1f}s"
            )

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_mape": val_metrics["mape"],
            }, step=epoch)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Test evaluation
        test_loss, test_preds, test_targets = evaluate(model, test_loader, criterion, device)

        # Desnormalitzar per obtenir mètriques en MW reals
        target_mean = scaler_params["mean"][args.target_col]
        target_std = scaler_params["std"][args.target_col]
        test_preds_mw = test_preds * target_std + target_mean
        test_targets_mw = test_targets * target_std + target_mean

        test_metrics_norm = compute_metrics(test_preds, test_targets)
        test_metrics_mw = compute_metrics(test_preds_mw, test_targets_mw)

        print("\n--- Test Results (normalised) ---")
        print(f"  Loss: {test_loss:.6f}")
        print(f"  MAE:  {test_metrics_norm['mae']:.4f}")
        print(f"  RMSE: {test_metrics_norm['rmse']:.4f}")
        print(f"  MAPE: {test_metrics_norm['mape']:.2f}%")

        print("\n--- Test Results (MW) ---")
        print(f"  MAE:  {test_metrics_mw['mae']:.1f} MW")
        print(f"  RMSE: {test_metrics_mw['rmse']:.1f} MW")
        print(f"  MAPE: {test_metrics_mw['mape']:.2f}%")

        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_mae_norm": test_metrics_norm["mae"],
            "test_rmse_norm": test_metrics_norm["rmse"],
            "test_mape": test_metrics_norm["mape"],
            "test_mae_mw": test_metrics_mw["mae"],
            "test_rmse_mw": test_metrics_mw["rmse"],
        })

        # Save model
        out_dir = root / "output" / "models"
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / f"mlp_baseline_seed{args.seed}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "scaler_params": scaler_params,
            "test_metrics_mw": test_metrics_mw,
        }, model_path)
        mlflow.log_artifact(str(model_path))
        print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
