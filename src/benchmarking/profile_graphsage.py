"""Profile the GraphSAGE benchmark on the unified feature set."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmarking.common import (
    ModelBenchmarkOutput,
    compute_metrics,
    current_rss_mb,
    cuda_peak_mb,
    model_size_mb,
    parameter_count,
    profile_torch_batches,
    save_json,
    trainable_parameter_count,
)
from src.data.graph_dataset import get_graph_dataloaders
from src.models.graphsage import GraphSAGEBaseline
from src.paths import METRICS_DIR, MODELS_DIR, ensure_artifact_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile the GraphSAGE benchmark")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pred_len", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128, 64])
    p.add_argument("--warmup_batches", type=int, default=3)
    p.add_argument("--timed_batches", type=int, default=20)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--output", type=Path, default=METRICS_DIR / "resource_benchmark" / "graphsage_seed42.json")
    return p.parse_args()


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model: nn.Module, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = criterion(pred[batch.source_mask], batch.y[batch.source_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device, mask_name: str):
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
    avg_loss = total_loss / max(n_batches, 1)
    preds = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
    targets = np.concatenate(all_targets, axis=0) if all_targets else np.array([])
    return avg_loss, preds, targets


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_artifact_dirs()
    print(f"[GraphSAGE] Starting profile | seed={args.seed} | epochs={args.epochs} | patience={args.patience}", flush=True)

    train_loader, val_loader, test_loader, feature_cols, target_cols, feature_params, target_params = get_graph_dataloaders(
        Path(ROOT),
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        include_temporal=True,
        include_weather=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphSAGEBaseline(
        input_dim=len(feature_cols),
        pred_len=args.pred_len,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    train_t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        source_val_loss, _, _ = evaluate(model, val_loader, criterion, device, "source_mask")
        if epoch == 1 or epoch % max(args.log_every, 1) == 0:
            print(
                f"[GraphSAGE] Epoch {epoch}/{args.epochs} | train_loss={train_loss:.6f} | source_val_loss={source_val_loss:.6f}",
                flush=True,
            )
        if source_val_loss < best_val_loss:
            best_val_loss = source_val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[GraphSAGE] Early stopping at epoch {epoch}", flush=True)
                break
    train_time_s = time.perf_counter() - train_t0
    print(f"[GraphSAGE] Training done in {train_time_s:.1f}s", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    source_train_loss, source_train_preds, source_train_targets = evaluate(model, train_loader, criterion, device, "source_mask")
    source_val_loss, source_val_preds, source_val_targets = evaluate(model, val_loader, criterion, device, "source_mask")
    source_test_loss, source_test_preds, source_test_targets = evaluate(model, test_loader, criterion, device, "source_mask")
    target_val_loss, target_val_preds, target_val_targets = evaluate(model, val_loader, criterion, device, "target_mask")
    target_test_loss, target_test_preds, target_test_targets = evaluate(model, test_loader, criterion, device, "target_mask")

    metrics = {
        "source_train": compute_metrics(source_train_targets, source_train_preds),
        "source_val": compute_metrics(source_val_targets, source_val_preds),
        "source_test": compute_metrics(source_test_targets, source_test_preds),
        "target_val": compute_metrics(target_val_targets, target_val_preds),
        "target_test": compute_metrics(target_test_targets, target_test_preds),
    }

    profile = {
        "source_test": profile_torch_batches(
            test_loader,
            lambda batch: model(batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device)).detach().cpu(),
            warmup_batches=args.warmup_batches,
            timed_batches=args.timed_batches,
        ),
        "target_test": profile_torch_batches(
            test_loader,
            lambda batch: model(batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device)).detach().cpu(),
            warmup_batches=args.warmup_batches,
            timed_batches=args.timed_batches,
        ),
    }

    model_path = MODELS_DIR / f"resource_graphsage_seed{args.seed}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "feature_cols": feature_cols,
            "target_cols": target_cols,
            "feature_params": feature_params,
            "target_params": target_params,
        },
        model_path,
    )

    output = ModelBenchmarkOutput(
        model_name="graphsage",
        seed=args.seed,
        feature_cols=feature_cols,
        target_cols=target_cols,
        n_parameters=parameter_count(model),
        n_trainable_parameters=trainable_parameter_count(model),
        model_size_mb=model_size_mb(model_path),
        peak_rss_mb=current_rss_mb(),
        peak_vram_mb=cuda_peak_mb(),
        train_time_s=train_time_s,
        fit_metrics=metrics,
        inference=profile,
        artifact_paths={"model": str(model_path)},
    )

    payload = output.to_dict()
    payload["losses"] = {
        "source_train": float(source_train_loss),
        "source_val": float(source_val_loss),
        "source_test": float(source_test_loss),
        "target_val": float(target_val_loss),
        "target_test": float(target_test_loss),
    }
    save_json(args.output, payload)
    print(f"[GraphSAGE] Saved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
