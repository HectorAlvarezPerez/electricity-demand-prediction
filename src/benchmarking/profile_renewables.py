"""Profile daily renewables models with and without external weather features."""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn
from xgboost import XGBRegressor

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
    profile_numpy_batches,
    profile_torch_batches,
    save_json,
    trainable_parameter_count,
)
from src.data.renewables_dataset import get_graph_dataloaders, get_tabular_dataloaders
from src.models.graphsage import GraphSAGEBaseline
from src.models.mlp_baseline import MLPBaseline
from src.paths import METRICS_DIR, MODELS_DIR, ROOT as PROJECT_ROOT, ensure_artifact_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile one daily renewables model")
    p.add_argument("--model", choices=["xgboost", "mlp", "graphsage"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include_external", action="store_true")
    p.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "data" / "processed_renewables_daily")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128, 64])
    p.add_argument("--n_estimators", type=int, default=100)
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample_bytree", type=float, default=0.8)
    p.add_argument("--n_jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--warmup_batches", type=int, default=3)
    p.add_argument("--timed_batches", type=int, default=20)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_xy(frame, x_cols: list[str], y_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return (
        frame[x_cols].to_numpy(dtype=np.float32, copy=True),
        frame[y_cols].to_numpy(dtype=np.float32, copy=True),
    )


def _param_array(values: Any, target_cols: list[str]) -> np.ndarray:
    if hasattr(values, "reindex"):
        values = values.reindex(target_cols)
    elif isinstance(values, dict):
        values = [values[col] for col in target_cols]
    elif hasattr(values, "to_numpy"):
        values = values.to_numpy()
    return np.asarray(values, dtype=np.float32)


def denormalize(values: np.ndarray, params: dict, target_cols: list[str]) -> np.ndarray:
    mean = _param_array(params["mean"], target_cols)
    std = _param_array(params["std"], target_cols)
    std = np.where(std == 0, 1.0, std)
    return np.asarray(values, dtype=np.float32) * std + mean


def metrics_with_raw(
    targets: np.ndarray,
    preds: np.ndarray,
    target_params: dict,
    y_cols: list[str],
) -> dict[str, dict[str, float]]:
    return {
        "norm": compute_metrics(targets, preds),
        "raw": compute_metrics(denormalize(targets, target_params, y_cols), denormalize(preds, target_params, y_cols)),
    }


def metrics_by_target(y_true: np.ndarray, y_pred: np.ndarray, y_cols: list[str]) -> dict[str, dict[str, float]]:
    return {
        col: compute_metrics(y_true[:, idx], y_pred[:, idx])
        for idx, col in enumerate(y_cols)
    }


def metrics_bundle(
    targets: np.ndarray,
    preds: np.ndarray,
    target_params: dict,
    y_cols: list[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    targets_raw = denormalize(targets, target_params, y_cols)
    preds_raw = denormalize(preds, target_params, y_cols)
    return (
        compute_metrics(targets, preds),
        compute_metrics(targets_raw, preds_raw),
        metrics_by_target(targets, preds, y_cols),
        metrics_by_target(targets_raw, preds_raw, y_cols),
    )


def train_xgboost(args: argparse.Namespace) -> dict:
    loaders, frames, x_cols, y_cols, feature_params, target_params = get_tabular_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        include_external=args.include_external,
        include_country_id=False,
    )
    del loaders, feature_params
    x_train, y_train = to_xy(frames["source_train"], x_cols, y_cols)
    x_source_val, y_source_val = to_xy(frames["source_val"], x_cols, y_cols)
    arrays = {
        name: to_xy(frame, x_cols, y_cols)
        for name, frame in frames.items()
    }
    model = XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        tree_method="hist",
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )
    t0 = time.perf_counter()
    model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_source_val, y_source_val)], verbose=False)
    train_time_s = time.perf_counter() - t0
    preds = {name: model.predict(x) for name, (x, _y) in arrays.items()}
    metrics, raw_metrics, by_target, by_target_raw = {}, {}, {}, {}
    for name, (_x, y) in arrays.items():
        metrics[name], raw_metrics[name], by_target[name], by_target_raw[name] = metrics_bundle(
            y, preds[name], target_params, y_cols
        )
    profile = {
        "source_test": profile_numpy_batches(
            arrays["source_test"][0],
            model.predict,
            batch_size=args.batch_size,
            warmup_batches=args.warmup_batches,
            timed_batches=args.timed_batches,
        ),
        "target_test": profile_numpy_batches(
            arrays["target_test"][0],
            model.predict,
            batch_size=args.batch_size,
            warmup_batches=args.warmup_batches,
            timed_batches=args.timed_batches,
        ),
    }
    feature_set = "external" if args.include_external else "no_external"
    model_path = MODELS_DIR / f"renewables_xgboost_{feature_set}_seed{args.seed}.joblib"
    joblib.dump(model, model_path)
    return build_payload(
        args, "xgboost", x_cols, y_cols, None, None, model_path, train_time_s, metrics, raw_metrics, by_target, by_target_raw, profile
    )


@torch.no_grad()
def evaluate_mlp(model: nn.Module, loader, criterion: nn.Module, device: torch.device):
    model.eval()
    losses = []
    preds, targets = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        losses.append(float(criterion(pred, y).item()))
        preds.append(pred.cpu().numpy())
        targets.append(y.cpu().numpy())
    return float(np.mean(losses)) if losses else 0.0, np.concatenate(preds), np.concatenate(targets)


def train_mlp(args: argparse.Namespace) -> dict:
    loaders, _frames, x_cols, y_cols, _feature_params, target_params = get_tabular_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        include_external=args.include_external,
        include_country_id=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPBaseline(input_dim=len(x_cols), pred_len=len(y_cols), hidden_dims=args.hidden_dims, dropout=args.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_state = None
    best_val = float("inf")
    patience = 0
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y in loaders["source_train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_loss, _pred, _target = evaluate_mlp(model, loaders["source_val"], criterion, device)
        if epoch == 1 or epoch % max(args.log_every, 1) == 0:
            print(f"[renewables/mlp] epoch={epoch} train_loss={np.mean(train_losses):.6f} source_val_loss={val_loss:.6f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    train_time_s = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    metrics, raw_metrics, by_target, by_target_raw = {}, {}, {}, {}
    for name, loader in loaders.items():
        _loss, pred, target = evaluate_mlp(model, loader, criterion, device)
        metrics[name], raw_metrics[name], by_target[name], by_target_raw[name] = metrics_bundle(
            target, pred, target_params, y_cols
        )

    def infer_batch(batch):
        x, _ = batch
        return model(x.to(device)).detach().cpu()

    profile = {
        "source_test": profile_torch_batches(loaders["source_test"], infer_batch, warmup_batches=args.warmup_batches, timed_batches=args.timed_batches),
        "target_test": profile_torch_batches(loaders["target_test"], infer_batch, warmup_batches=args.warmup_batches, timed_batches=args.timed_batches),
    }
    feature_set = "external" if args.include_external else "no_external"
    model_path = MODELS_DIR / f"renewables_mlp_{feature_set}_seed{args.seed}.pt"
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "feature_cols": x_cols, "target_cols": y_cols}, model_path)
    return build_payload(
        args,
        "mlp",
        x_cols,
        y_cols,
        parameter_count(model),
        trainable_parameter_count(model),
        model_path,
        train_time_s,
        metrics,
        raw_metrics,
        by_target,
        by_target_raw,
        profile,
    )


def train_graph_epoch(model: nn.Module, loader, optimizer, criterion, device) -> float:
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = criterion(pred[batch.source_mask], batch.y[batch.source_mask])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate_graph(model: nn.Module, loader, criterion: nn.Module, device: torch.device, mask_name: str):
    model.eval()
    losses, preds, targets = [], [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr)
        mask = getattr(batch, mask_name)
        if mask.sum() == 0:
            continue
        masked_pred = pred[mask]
        masked_target = batch.y[mask]
        losses.append(float(criterion(masked_pred, masked_target).item()))
        preds.append(masked_pred.cpu().numpy())
        targets.append(masked_target.cpu().numpy())
    return float(np.mean(losses)) if losses else 0.0, np.concatenate(preds), np.concatenate(targets)


def train_graphsage(args: argparse.Namespace) -> dict:
    loaders, x_cols, y_cols, _feature_params, target_params = get_graph_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        include_external=args.include_external,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphSAGEBaseline(input_dim=len(x_cols), pred_len=len(y_cols), hidden_dims=args.hidden_dims, dropout=args.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_state = None
    best_val = float("inf")
    patience = 0
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_graph_epoch(model, loaders["source_train"], optimizer, criterion, device)
        source_val_loss, _pred, _target = evaluate_graph(model, loaders["val"], criterion, device, "source_mask")
        if epoch == 1 or epoch % max(args.log_every, 1) == 0:
            print(f"[renewables/graphsage] epoch={epoch} train_loss={train_loss:.6f} source_val_loss={source_val_loss:.6f}", flush=True)
        if source_val_loss < best_val:
            best_val = source_val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    train_time_s = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    eval_specs = {
        "source_train": (loaders["source_train"], "source_mask"),
        "source_val": (loaders["val"], "source_mask"),
        "source_test": (loaders["test"], "source_mask"),
        "target_val": (loaders["val"], "target_mask"),
        "target_test": (loaders["test"], "target_mask"),
    }
    metrics, raw_metrics, by_target, by_target_raw = {}, {}, {}, {}
    for name, (loader, mask_name) in eval_specs.items():
        _loss, pred, target = evaluate_graph(model, loader, criterion, device, mask_name)
        metrics[name], raw_metrics[name], by_target[name], by_target_raw[name] = metrics_bundle(
            target, pred, target_params, y_cols
        )

    def infer_graph(batch):
        batch = batch.to(device)
        return model(batch.x, batch.edge_index, batch.edge_attr).detach().cpu()

    profile = {
        "source_test": profile_torch_batches(loaders["test"], infer_graph, warmup_batches=args.warmup_batches, timed_batches=args.timed_batches),
        "target_test": profile_torch_batches(loaders["test"], infer_graph, warmup_batches=args.warmup_batches, timed_batches=args.timed_batches),
    }
    feature_set = "external" if args.include_external else "no_external"
    model_path = MODELS_DIR / f"renewables_graphsage_{feature_set}_seed{args.seed}.pt"
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "feature_cols": x_cols, "target_cols": y_cols}, model_path)
    return build_payload(
        args,
        "graphsage",
        x_cols,
        y_cols,
        parameter_count(model),
        trainable_parameter_count(model),
        model_path,
        train_time_s,
        metrics,
        raw_metrics,
        by_target,
        by_target_raw,
        profile,
    )


def build_payload(
    args: argparse.Namespace,
    model_name: str,
    x_cols: list[str],
    y_cols: list[str],
    n_parameters: int | None,
    n_trainable_parameters: int | None,
    model_path: Path,
    train_time_s: float,
    metrics: dict[str, dict[str, float]],
    raw_metrics: dict[str, dict[str, float]],
    by_target: dict[str, dict[str, dict[str, float]]],
    by_target_raw: dict[str, dict[str, dict[str, float]]],
    profile: dict[str, dict[str, float]],
) -> dict:
    output = ModelBenchmarkOutput(
        model_name=model_name,
        seed=args.seed,
        feature_cols=x_cols,
        target_cols=y_cols,
        n_parameters=n_parameters,
        n_trainable_parameters=n_trainable_parameters,
        model_size_mb=model_size_mb(model_path),
        peak_rss_mb=current_rss_mb(),
        peak_vram_mb=cuda_peak_mb(),
        train_time_s=train_time_s,
        fit_metrics=metrics,
        inference=profile,
        artifact_paths={"model": str(model_path)},
    )
    payload = output.to_dict()
    payload["benchmark"] = "renewables_daily"
    payload["feature_set"] = "external" if args.include_external else "no_external"
    payload["fit_metrics_raw"] = raw_metrics
    payload["fit_metrics_by_target"] = by_target
    payload["fit_metrics_by_target_raw"] = by_target_raw
    return payload


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_artifact_dirs()
    feature_set = "external" if args.include_external else "no_external"
    if args.output is None:
        args.output = METRICS_DIR / "renewables_resource_benchmark" / f"{args.model}_{feature_set}_seed{args.seed}.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[renewables] model={args.model} feature_set={feature_set} data_dir={args.data_dir}", flush=True)
    if args.model == "xgboost":
        payload = train_xgboost(args)
    elif args.model == "mlp":
        payload = train_mlp(args)
    else:
        payload = train_graphsage(args)
    save_json(args.output, payload)
    print(f"[renewables] Saved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
