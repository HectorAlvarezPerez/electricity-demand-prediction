"""Profile the tabular MLP benchmark on the unified feature set."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmarking.common import (
    ModelBenchmarkOutput,
    build_conformal_interval_report,
    build_prediction_interval_frame,
    compute_metrics,
    current_rss_mb,
    cuda_peak_mb,
    model_size_mb,
    parameter_count,
    profile_torch_batches,
    save_json,
    trainable_parameter_count,
)
from src.data.dataset import create_dataloader
from src.data.preprocess import feature_columns, normalize_data, target_columns
from src.models.mlp_baseline import MLPBaseline
from src.paths import METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, ensure_artifact_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile the MLP benchmark")
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
    p.add_argument("--output", type=Path, default=METRICS_DIR / "resource_benchmark" / "mlp_seed42.json")
    return p.parse_args()


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split(path: Path):
    import pandas as pd

    return pd.read_parquet(path)


def split_for_roles(df, roles: set[str]):
    return df[df["role"].isin(roles)].reset_index(drop=True)


def scale_frames(train_df, val_df, test_df, feature_cols, target_cols):
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


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device):
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


def train_one_epoch(model: nn.Module, loader, optimizer, criterion, device) -> float:
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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_artifact_dirs()
    print(f"[MLP] Starting profile | seed={args.seed} | epochs={args.epochs} | patience={args.patience}", flush=True)

    train_df = load_split(PROCESSED_DATA_DIR / "train.parquet")
    val_df = load_split(PROCESSED_DATA_DIR / "val.parquet")
    test_df = load_split(PROCESSED_DATA_DIR / "test.parquet")

    train_df = split_for_roles(train_df, {"source"})
    source_val_df = split_for_roles(val_df, {"source"})
    source_test_df = split_for_roles(test_df, {"source"})
    target_val_df = split_for_roles(val_df, {"target"})
    target_test_df = split_for_roles(test_df, {"target"})

    y_cols = target_columns(args.pred_len)
    x_cols = feature_columns(
        train_df,
        include_temporal=True,
        include_weather=True,
        include_country_id=False,
    )

    train_scaled, source_val_scaled, source_test_scaled, feature_params, target_params = scale_frames(
        train_df, source_val_df, source_test_df, x_cols, y_cols
    )
    target_val_scaled = target_val_df.copy()
    target_test_scaled = target_test_df.copy()
    target_val_features, _ = normalize_data(target_val_scaled[x_cols], method="standard", params=feature_params)
    target_test_features, _ = normalize_data(target_test_scaled[x_cols], method="standard", params=feature_params)
    target_val_targets, _ = normalize_data(target_val_scaled[y_cols], method="standard", params=target_params)
    target_test_targets, _ = normalize_data(target_test_scaled[y_cols], method="standard", params=target_params)
    target_val_scaled[x_cols] = target_val_features
    target_test_scaled[x_cols] = target_test_features
    target_val_scaled[y_cols] = target_val_targets
    target_test_scaled[y_cols] = target_test_targets

    train_loader = create_dataloader(train_scaled, x_cols, y_cols, batch_size=args.batch_size, shuffle=True)
    source_val_loader = create_dataloader(source_val_scaled, x_cols, y_cols, batch_size=args.batch_size, shuffle=False)
    source_test_loader = create_dataloader(source_test_scaled, x_cols, y_cols, batch_size=args.batch_size, shuffle=False)
    target_val_loader = create_dataloader(target_val_scaled, x_cols, y_cols, batch_size=args.batch_size, shuffle=False)
    target_test_loader = create_dataloader(target_test_scaled, x_cols, y_cols, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPBaseline(
        input_dim=len(x_cols),
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
        source_val_loss, _, _ = evaluate(model, source_val_loader, criterion, device)
        if epoch == 1 or epoch % max(args.log_every, 1) == 0:
            print(
                f"[MLP] Epoch {epoch}/{args.epochs} | train_loss={train_loss:.6f} | source_val_loss={source_val_loss:.6f}",
                flush=True,
            )
        if source_val_loss < best_val_loss:
            best_val_loss = source_val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[MLP] Early stopping at epoch {epoch}", flush=True)
                break
    train_time_s = time.perf_counter() - train_t0
    print(f"[MLP] Training done in {train_time_s:.1f}s", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    source_train_loss, source_train_preds, source_train_targets = evaluate(model, train_loader, criterion, device)
    source_val_loss, source_val_preds, source_val_targets = evaluate(model, source_val_loader, criterion, device)
    source_test_loss, source_test_preds, source_test_targets = evaluate(model, source_test_loader, criterion, device)
    target_val_loss, target_val_preds, target_val_targets = evaluate(model, target_val_loader, criterion, device)
    target_test_loss, target_test_preds, target_test_targets = evaluate(model, target_test_loader, criterion, device)

    metrics = {
        "source_train": compute_metrics(source_train_targets, source_train_preds),
        "source_val": compute_metrics(source_val_targets, source_val_preds),
        "source_test": compute_metrics(source_test_targets, source_test_preds),
        "target_val": compute_metrics(target_val_targets, target_val_preds),
        "target_test": compute_metrics(target_test_targets, target_test_preds),
    }

    prediction_intervals = build_conformal_interval_report(
        calibrations={
            "source_val": (source_val_targets, source_val_preds),
            "target_val": (target_val_targets, target_val_preds),
        },
        evaluations={
            "source_test": (source_test_targets, source_test_preds),
            "target_test": (target_test_targets, target_test_preds),
        },
        target_params=target_params,
        target_cols=y_cols,
        alpha=0.05,
    )
    interval_frames = []
    for calibration_name, calibration_payload in prediction_intervals["calibrations"].items():
        quantiles_norm = np.asarray(calibration_payload["quantiles_norm"], dtype=np.float32)
        for split_name, frame, y_true, y_pred in [
            ("source_test", source_test_scaled, source_test_targets, source_test_preds),
            ("target_test", target_test_scaled, target_test_targets, target_test_preds),
        ]:
            interval_frames.append(
                build_prediction_interval_frame(
                    model_name="mlp",
                    split_name=split_name,
                    calibration_name=calibration_name,
                    metadata=frame,
                    y_true_norm=y_true,
                    y_pred_norm=y_pred,
                    quantiles_norm=quantiles_norm,
                    target_params=target_params,
                    target_cols=y_cols,
                    alpha=0.05,
                )
            )

    def infer_batch(batch_data: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, _ = batch_data
        return model(x.to(device)).detach().cpu()

    profile = {
        "source_test": profile_torch_batches(
            source_test_loader,
            infer_batch,
            warmup_batches=args.warmup_batches,
            timed_batches=args.timed_batches,
        ),
        "target_test": profile_torch_batches(
            target_test_loader,
            infer_batch,
            warmup_batches=args.warmup_batches,
            timed_batches=args.timed_batches,
        ),
    }

    model_path = MODELS_DIR / f"resource_mlp_seed{args.seed}.pt"
    intervals_path = args.output.with_name(f"{args.output.stem}_prediction_intervals.parquet")
    pd.concat(interval_frames, ignore_index=True).to_parquet(intervals_path, index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "feature_cols": x_cols,
            "target_cols": y_cols,
            "feature_params": feature_params,
            "target_params": target_params,
        },
        model_path,
    )

    output = ModelBenchmarkOutput(
        model_name="mlp",
        seed=args.seed,
        feature_cols=x_cols,
        target_cols=y_cols,
        n_parameters=parameter_count(model),
        n_trainable_parameters=trainable_parameter_count(model),
        model_size_mb=model_size_mb(model_path),
        peak_rss_mb=current_rss_mb(),
        peak_vram_mb=cuda_peak_mb(),
        train_time_s=train_time_s,
        fit_metrics=metrics,
        inference=profile,
        artifact_paths={"model": str(model_path), "prediction_intervals": str(intervals_path)},
    )

    # Add the extra scalar losses for traceability.
    payload = output.to_dict()
    payload["losses"] = {
        "source_train": float(source_train_loss),
        "source_val": float(source_val_loss),
        "source_test": float(source_test_loss),
        "target_val": float(target_val_loss),
        "target_test": float(target_test_loss),
    }
    payload["prediction_intervals"] = prediction_intervals
    save_json(args.output, payload)
    print(f"[MLP] Saved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
