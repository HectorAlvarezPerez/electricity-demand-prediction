"""
Training loop for the GraphSAGE explicit-lag forecasting pipeline.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.graph_dataset import get_graph_dataloaders
from src.models.graphsage import GraphSAGEBaseline
from src.paths import METRICS_DIR, MODELS_DIR, ensure_artifact_dirs

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    preds = preds.flatten()
    targets = targets.flatten()
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mask = np.abs(targets) > 1e-3
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((preds[mask] - targets[mask]) / targets[mask])) * 100)
    else:
        mape = float("nan")
    return {"mae": mae, "rmse": rmse, "mape": mape}


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.edge_attr)
        
        # Supervise only on source nodes during training!
        loss = criterion(pred[batch.source_mask], batch.y[batch.source_mask])
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device, mask_name: str):
    """
    Evaluate the model on a specific mask: 'source_mask' or 'target_mask'.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds, all_targets = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr)
        
        mask = getattr(batch, mask_name)
        if mask.sum() == 0:
            continue
            
        masked_pred = pred[mask]
        masked_y = batch.y[mask]
        
        loss = criterion(masked_pred, masked_y)
        total_loss += loss.item()
        n_batches += 1
        all_preds.append(masked_pred.cpu().numpy())
        all_targets.append(masked_y.cpu().numpy())
        
    if n_batches == 0:
        return 0.0, np.array([]), np.array([])
        
    avg_loss = total_loss / n_batches
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return avg_loss, preds, targets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GraphSAGE forecasting model")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pred_len", type=int, default=24, help="Forecast horizon (hours)")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128, 64])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--include_temporal", dest="include_temporal", action="store_true")
    p.add_argument("--no-include-temporal", dest="include_temporal", action="store_false")
    p.add_argument("--include_weather", dest="include_weather", action="store_true")
    p.add_argument("--no-include-weather", dest="include_weather", action="store_false")
    p.add_argument("--experiment_name", type=str, default="gnn_tabular_long")
    p.set_defaults(
        include_temporal=True,
        include_weather=True,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading graph-format processed data...")
    (
        train_loader,
        val_loader,
        test_loader,
        feature_cols,
        target_cols,
        feature_params,
        target_params,
    ) = get_graph_dataloaders(
        ROOT,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        include_temporal=args.include_temporal,
        include_weather=args.include_weather,
    )

    model = GraphSAGEBaseline(
        input_dim=len(feature_cols),
        pred_len=args.pred_len,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)
    print(f"Input features per node: {len(feature_cols)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run():
        run = mlflow.active_run()
        mlflow.log_params(vars(args))
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("device", str(device))

        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        best_model_state = None
        history = []

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            
            source_val_loss, source_val_preds, source_val_targets = evaluate(
                model, val_loader, criterion, device, "source_mask"
            )
            target_val_loss, target_val_preds, target_val_targets = evaluate(
                model, val_loader, criterion, device, "target_mask"
            )
            
            elapsed = time.time() - t0
            source_val_metrics = compute_metrics(source_val_preds, source_val_targets)
            target_val_metrics = compute_metrics(target_val_preds, target_val_targets)

            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"train_loss={train_loss:.6f} | "
                f"source_val_loss={source_val_loss:.6f} | "
                f"source_val_mae={source_val_metrics['mae']:.4f} | "
                f"target_val_mae={target_val_metrics['mae']:.4f} | "
                f"{elapsed:.1f}s"
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "source_val_loss": source_val_loss,
                    "source_val_mae": source_val_metrics["mae"],
                    "source_val_rmse": source_val_metrics["rmse"],
                    "source_val_mape": source_val_metrics["mape"],
                    "target_val_loss": target_val_loss,
                    "target_val_mae": target_val_metrics["mae"],
                    "target_val_rmse": target_val_metrics["rmse"],
                    "target_val_mape": target_val_metrics["mape"],
                },
                step=epoch,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "source_val_loss": source_val_loss,
                    "source_val_mae": source_val_metrics["mae"],
                    "source_val_rmse": source_val_metrics["rmse"],
                    "target_val_loss": target_val_loss,
                    "target_val_mae": target_val_metrics["mae"],
                    "target_val_rmse": target_val_metrics["rmse"],
                }
            )

            # We select model based on source validation
            if source_val_loss < best_val_loss:
                best_val_loss = source_val_loss
                best_epoch = epoch
                patience_counter = 0
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        source_test_loss, source_test_preds, source_test_targets = evaluate(
            model, test_loader, criterion, device, "source_mask"
        )
        target_test_loss, target_test_preds, target_test_targets = evaluate(
            model, test_loader, criterion, device, "target_mask"
        )
        source_test_metrics_norm = compute_metrics(source_test_preds, source_test_targets)
        target_test_metrics_norm = compute_metrics(target_test_preds, target_test_targets)

        mean = target_params["mean"].to_numpy(dtype=np.float32)
        std = target_params["std"].replace(0, 1).to_numpy(dtype=np.float32)
        source_test_preds_mw = source_test_preds * std + mean
        source_test_targets_mw = source_test_targets * std + mean
        target_test_preds_mw = target_test_preds * std + mean
        target_test_targets_mw = target_test_targets * std + mean
        source_test_metrics_mw = compute_metrics(source_test_preds_mw, source_test_targets_mw)
        target_test_metrics_mw = compute_metrics(target_test_preds_mw, target_test_targets_mw)

        print("\n--- Source Test Results (normalised) ---")
        print(f"  Loss: {source_test_loss:.6f}")
        print(f"  MAE:  {source_test_metrics_norm['mae']:.4f}")
        print(f"  RMSE: {source_test_metrics_norm['rmse']:.4f}")
        print(f"  MAPE: {source_test_metrics_norm['mape']:.2f}%")

        print("\n--- Source Test Results (MW) ---")
        print(f"  MAE:  {source_test_metrics_mw['mae']:.1f} MW")
        print(f"  RMSE: {source_test_metrics_mw['rmse']:.1f} MW")
        print(f"  MAPE: {source_test_metrics_mw['mape']:.2f}%")

        print("\n--- Target Validation Results (normalised) ---")
        print(f"  Loss: {target_val_loss:.6f}")
        print(f"  MAE:  {target_val_metrics['mae']:.4f}")
        print(f"  RMSE: {target_val_metrics['rmse']:.4f}")
        print(f"  MAPE: {target_val_metrics['mape']:.2f}%")

        print("\n--- Target Test Results (normalised) ---")
        print(f"  Loss: {target_test_loss:.6f}")
        print(f"  MAE:  {target_test_metrics_norm['mae']:.4f}")
        print(f"  RMSE: {target_test_metrics_norm['rmse']:.4f}")
        print(f"  MAPE: {target_test_metrics_norm['mape']:.2f}%")

        print("\n--- Target Test Results (MW) ---")
        print(f"  MAE:  {target_test_metrics_mw['mae']:.1f} MW")
        print(f"  RMSE: {target_test_metrics_mw['rmse']:.1f} MW")
        print(f"  MAPE: {target_test_metrics_mw['mape']:.2f}%")

        mlflow.log_metrics(
            {
                "source_test_loss": source_test_loss,
                "source_test_mae_norm": source_test_metrics_norm["mae"],
                "source_test_rmse_norm": source_test_metrics_norm["rmse"],
                "source_test_mape": source_test_metrics_norm["mape"],
                "source_test_mae_mw": source_test_metrics_mw["mae"],
                "source_test_rmse_mw": source_test_metrics_mw["rmse"],
                "target_test_loss": target_test_loss,
                "target_test_mae_norm": target_test_metrics_norm["mae"],
                "target_test_rmse_norm": target_test_metrics_norm["rmse"],
                "target_test_mape": target_test_metrics_norm["mape"],
                "target_test_mae_mw": target_test_metrics_mw["mae"],
                "target_test_rmse_mw": target_test_metrics_mw["rmse"],
            }
        )

        ensure_artifact_dirs()
        model_path = MODELS_DIR / f"gnn_tabular_long_seed{args.seed}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "feature_cols": feature_cols,
                "target_cols": target_cols,
                "feature_params": feature_params,
                "target_params": target_params,
                "source_test_metrics_mw": source_test_metrics_mw,
                "target_test_metrics_mw": target_test_metrics_mw,
            },
            model_path,
        )
        mlflow.log_artifact(str(model_path))

        metrics_path = METRICS_DIR / f"gnn_metrics_seed{args.seed}.json"
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "run_id": run.info.run_id if run is not None else None,
                    "seed": args.seed,
                    "n_features": len(feature_cols),
                    "n_parameters": int(sum(p.numel() for p in model.parameters())),
                    "best_epoch": best_epoch,
                    "best_source_val_loss": best_val_loss,
                    "history": history,
                    "source_test": {
                        "loss_norm": source_test_loss,
                        "metrics_norm": source_test_metrics_norm,
                        "metrics_mw": source_test_metrics_mw,
                    },
                    "target_val": {
                        "loss_norm": target_val_loss,
                        "metrics_norm": target_val_metrics,
                    },
                    "target_test": {
                        "loss_norm": target_test_loss,
                        "metrics_norm": target_test_metrics_norm,
                        "metrics_mw": target_test_metrics_mw,
                    },
                },
                f,
                indent=2,
            )
        mlflow.log_artifact(str(metrics_path))
        print(f"\nModel saved to {model_path}")
        print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
