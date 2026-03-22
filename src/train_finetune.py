"""
Few-shot fine-tuning for the tabular MLP forecasting pipeline.

Workflow:
  1. Load a source-pretrained MLP checkpoint from src/train.py
  2. Select a small fraction of target-domain training samples
  3. Fine-tune on target only, validating on target_val
  4. Evaluate on target_test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import create_dataloader
from src.data.preprocess import normalize_data
from src.models.mlp_baseline import MLPBaseline
from src.paths import METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, ensure_artifact_dirs
from src.train import compute_metrics, evaluate, set_seed, train_one_epoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune a pretrained MLP on the target domain")
    p.add_argument(
        "--pretrained_model",
        type=Path,
        default=MODELS_DIR / "mlp_tabular_long_seed42.pt",
        help="Checkpoint produced by src/train.py",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--target_fraction", type=float, default=0.05)
    p.add_argument(
        "--sampling",
        choices=["head", "random"],
        default="head",
        help="How to pick the few-shot target subset",
    )
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--experiment_name", type=str, default="mlp_target_finetune")
    return p.parse_args()


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def denormalize_targets(values: np.ndarray, target_params: dict) -> np.ndarray:
    mean = target_params["mean"].to_numpy(dtype=np.float32)
    std = target_params["std"].replace(0, 1).to_numpy(dtype=np.float32)
    return values * std + mean


def sample_target_train(df: pd.DataFrame, fraction: float, sampling: str, seed: int) -> pd.DataFrame:
    if not 0 < fraction <= 1:
        raise ValueError("--target_fraction must be in (0, 1].")

    df = df.sort_values("utc_timestamp").reset_index(drop=True)
    n_samples = max(1, int(np.ceil(len(df) * fraction)))

    if sampling == "head":
        sampled = df.iloc[:n_samples].copy()
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(df), size=n_samples, replace=False))
        sampled = df.iloc[idx].copy()

    return sampled.reset_index(drop=True)


def build_target_loaders(
    root: Path,
    feature_cols: list[str],
    target_cols: list[str],
    feature_params: dict,
    target_params: dict,
    *,
    batch_size: int,
    target_fraction: float,
    sampling: str,
    seed: int,
):
    train_all = load_split(PROCESSED_DATA_DIR / "train.parquet")
    val_all = load_split(PROCESSED_DATA_DIR / "val.parquet")
    test_all = load_split(PROCESSED_DATA_DIR / "test.parquet")

    target_train_full = train_all[train_all["role"] == "target"].reset_index(drop=True)
    target_val = val_all[val_all["role"] == "target"].reset_index(drop=True)
    target_test = test_all[test_all["role"] == "target"].reset_index(drop=True)

    target_train = sample_target_train(target_train_full, target_fraction, sampling, seed)

    train_features, _ = normalize_data(target_train[feature_cols], method="standard", params=feature_params)
    val_features, _ = normalize_data(target_val[feature_cols], method="standard", params=feature_params)
    test_features, _ = normalize_data(target_test[feature_cols], method="standard", params=feature_params)

    train_targets, _ = normalize_data(target_train[target_cols], method="standard", params=target_params)
    val_targets, _ = normalize_data(target_val[target_cols], method="standard", params=target_params)
    test_targets, _ = normalize_data(target_test[target_cols], method="standard", params=target_params)

    target_train = target_train.copy()
    target_val = target_val.copy()
    target_test = target_test.copy()
    target_train[feature_cols] = train_features
    target_val[feature_cols] = val_features
    target_test[feature_cols] = test_features
    target_train[target_cols] = train_targets
    target_val[target_cols] = val_targets
    target_test[target_cols] = test_targets

    train_loader = create_dataloader(target_train, feature_cols, target_cols, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(target_val, feature_cols, target_cols, batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader(target_test, feature_cols, target_cols, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, len(target_train), len(target_train_full)


def set_trainable_parameters(model: MLPBaseline, freeze_backbone: bool) -> None:
    for param in model.parameters():
        param.requires_grad = True

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        final_layer = model.net[-1]
        if not isinstance(final_layer, nn.Linear):
            raise TypeError("Expected the last MLP module to be nn.Linear.")
        final_layer.weight.requires_grad = True
        final_layer.bias.requires_grad = True


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not args.pretrained_model.exists():
        raise FileNotFoundError(args.pretrained_model)

    # This checkpoint is produced locally by src/train.py and stores
    # metadata beyond raw tensors (for example pandas-backed scaling stats),
    # so PyTorch 2.6+ must load it with weights_only disabled.
    checkpoint = torch.load(args.pretrained_model, map_location="cpu", weights_only=False)
    pretrained_args = checkpoint["args"]
    feature_cols = checkpoint["feature_cols"]
    target_cols = checkpoint["target_cols"]
    feature_params = checkpoint["feature_params"]
    target_params = checkpoint["target_params"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading pretrained checkpoint: {args.pretrained_model}")

    train_loader, val_loader, test_loader, n_target_samples, n_target_full = build_target_loaders(
        ROOT,
        feature_cols,
        target_cols,
        feature_params,
        target_params,
        batch_size=args.batch_size,
        target_fraction=args.target_fraction,
        sampling=args.sampling,
        seed=args.seed,
    )
    print(f"Target few-shot subset: {n_target_samples}/{n_target_full} samples ({args.target_fraction:.1%})")

    model = MLPBaseline(
        input_dim=len(feature_cols),
        pred_len=pretrained_args["pred_len"],
        hidden_dims=pretrained_args["hidden_dims"],
        dropout=pretrained_args["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    set_trainable_parameters(model, args.freeze_backbone)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    zero_val_loss, zero_val_preds, zero_val_targets = evaluate(model, val_loader, criterion, device)
    zero_test_loss, zero_test_preds, zero_test_targets = evaluate(model, test_loader, criterion, device)
    zero_val_metrics = compute_metrics(zero_val_preds, zero_val_targets)
    zero_test_metrics = compute_metrics(zero_test_preds, zero_test_targets)
    zero_test_preds_mw = denormalize_targets(zero_test_preds, target_params)
    zero_test_targets_mw = denormalize_targets(zero_test_targets, target_params)
    zero_test_metrics_mw = compute_metrics(zero_test_preds_mw, zero_test_targets_mw)

    print("\n--- Zero-shot target performance before fine-tuning ---")
    print(f"  Val MAE:  {zero_val_metrics['mae']:.4f}")
    print(f"  Test MAE: {zero_test_metrics['mae']:.4f}")
    print(f"  Test RMSE:{zero_test_metrics['rmse']:.4f}")
    print(f"  Test MAE: {zero_test_metrics_mw['mae']:.1f} MW")
    print(f"  Test RMSE:{zero_test_metrics_mw['rmse']:.1f} MW")

    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run():
        run = mlflow.active_run()
        mlflow.log_params(
            {
                **vars(args),
                "n_features": len(feature_cols),
                "pretrained_seed": pretrained_args["seed"],
                "pretrained_model": str(args.pretrained_model),
                "target_samples": n_target_samples,
                "target_samples_full": n_target_full,
            }
        )

        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        best_model_state = None
        history = []

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
            elapsed = time.time() - t0
            val_metrics = compute_metrics(val_preds, val_targets)

            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"train_loss={train_loss:.6f} | "
                f"target_val_loss={val_loss:.6f} | "
                f"target_val_mae={val_metrics['mae']:.4f} | "
                f"{elapsed:.1f}s"
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "target_val_loss": val_loss,
                    "target_val_mae": val_metrics["mae"],
                    "target_val_rmse": val_metrics["rmse"],
                    "target_val_mape": val_metrics["mape"],
                },
                step=epoch,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "target_val_loss": val_loss,
                    "target_val_mae": val_metrics["mae"],
                    "target_val_rmse": val_metrics["rmse"],
                }
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
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

        target_test_loss, target_test_preds, target_test_targets = evaluate(model, test_loader, criterion, device)
        target_test_metrics_norm = compute_metrics(target_test_preds, target_test_targets)

        target_test_preds_mw = denormalize_targets(target_test_preds, target_params)
        target_test_targets_mw = denormalize_targets(target_test_targets, target_params)
        target_test_metrics_mw = compute_metrics(target_test_preds_mw, target_test_targets_mw)
        improvement = {
            "mae_norm_abs": zero_test_metrics["mae"] - target_test_metrics_norm["mae"],
            "rmse_norm_abs": zero_test_metrics["rmse"] - target_test_metrics_norm["rmse"],
            "mae_mw_abs": zero_test_metrics_mw["mae"] - target_test_metrics_mw["mae"],
            "rmse_mw_abs": zero_test_metrics_mw["rmse"] - target_test_metrics_mw["rmse"],
        }
        improvement["mae_norm_pct"] = (
            improvement["mae_norm_abs"] / zero_test_metrics["mae"] * 100 if zero_test_metrics["mae"] else float("nan")
        )
        improvement["rmse_norm_pct"] = (
            improvement["rmse_norm_abs"] / zero_test_metrics["rmse"] * 100 if zero_test_metrics["rmse"] else float("nan")
        )
        improvement["mae_mw_pct"] = (
            improvement["mae_mw_abs"] / zero_test_metrics_mw["mae"] * 100 if zero_test_metrics_mw["mae"] else float("nan")
        )
        improvement["rmse_mw_pct"] = (
            improvement["rmse_mw_abs"] / zero_test_metrics_mw["rmse"] * 100 if zero_test_metrics_mw["rmse"] else float("nan")
        )

        print("\n--- Fine-tuned target test results (normalised) ---")
        print(f"  Loss: {target_test_loss:.6f}")
        print(f"  MAE:  {target_test_metrics_norm['mae']:.4f}")
        print(f"  RMSE: {target_test_metrics_norm['rmse']:.4f}")
        print(f"  MAPE: {target_test_metrics_norm['mape']:.2f}%")

        print("\n--- Fine-tuned target test results (MW) ---")
        print(f"  MAE:  {target_test_metrics_mw['mae']:.1f} MW")
        print(f"  RMSE: {target_test_metrics_mw['rmse']:.1f} MW")
        print(f"  MAPE: {target_test_metrics_mw['mape']:.2f}%")

        print("\n--- Improvement vs zero-shot target test ---")
        print(f"  Delta MAE (norm): {improvement['mae_norm_abs']:+.4f} ({improvement['mae_norm_pct']:+.2f}%)")
        print(f"  Delta RMSE (norm): {improvement['rmse_norm_abs']:+.4f} ({improvement['rmse_norm_pct']:+.2f}%)")
        print(f"  Delta MAE (MW):   {improvement['mae_mw_abs']:+.1f} MW ({improvement['mae_mw_pct']:+.2f}%)")
        print(f"  Delta RMSE (MW):  {improvement['rmse_mw_abs']:+.1f} MW ({improvement['rmse_mw_pct']:+.2f}%)")

        mlflow.log_metrics(
            {
                "zero_shot_target_val_loss": zero_val_loss,
                "zero_shot_target_val_mae": zero_val_metrics["mae"],
                "zero_shot_target_val_rmse": zero_val_metrics["rmse"],
                "zero_shot_target_test_loss": zero_test_loss,
                "zero_shot_target_test_mae": zero_test_metrics["mae"],
                "zero_shot_target_test_rmse": zero_test_metrics["rmse"],
                "zero_shot_target_test_mae_mw": zero_test_metrics_mw["mae"],
                "zero_shot_target_test_rmse_mw": zero_test_metrics_mw["rmse"],
                "finetune_target_test_loss": target_test_loss,
                "finetune_target_test_mae_norm": target_test_metrics_norm["mae"],
                "finetune_target_test_rmse_norm": target_test_metrics_norm["rmse"],
                "finetune_target_test_mae_mw": target_test_metrics_mw["mae"],
                "finetune_target_test_rmse_mw": target_test_metrics_mw["rmse"],
                "improvement_target_test_mae_norm_abs": improvement["mae_norm_abs"],
                "improvement_target_test_rmse_norm_abs": improvement["rmse_norm_abs"],
                "improvement_target_test_mae_mw_abs": improvement["mae_mw_abs"],
                "improvement_target_test_rmse_mw_abs": improvement["rmse_mw_abs"],
            }
        )

        suffix = f"seed{args.seed}_frac{args.target_fraction:.3f}".replace(".", "p")
        if args.freeze_backbone:
            suffix += "_headonly"

        ensure_artifact_dirs()
        model_path = MODELS_DIR / f"mlp_finetune_{suffix}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "feature_cols": feature_cols,
                "target_cols": target_cols,
                "feature_params": feature_params,
                "target_params": target_params,
                "best_epoch": best_epoch,
                "target_fraction": args.target_fraction,
                "freeze_backbone": args.freeze_backbone,
                "target_test_metrics_norm": target_test_metrics_norm,
                "target_test_metrics_mw": target_test_metrics_mw,
            },
            model_path,
        )

        metrics_path = METRICS_DIR / f"mlp_finetune_{suffix}.json"
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "run_id": run.info.run_id if run is not None else None,
                    "pretrained_model": str(args.pretrained_model),
                    "target_fraction": args.target_fraction,
                    "sampling": args.sampling,
                    "freeze_backbone": args.freeze_backbone,
                    "target_samples": n_target_samples,
                    "target_samples_full": n_target_full,
                    "best_epoch": best_epoch,
                    "best_target_val_loss": best_val_loss,
                    "history": history,
                    "zero_shot_target_val": {
                        "loss_norm": zero_val_loss,
                        "metrics_norm": zero_val_metrics,
                    },
                    "zero_shot_target_test": {
                        "loss_norm": zero_test_loss,
                        "metrics_norm": zero_test_metrics,
                        "metrics_mw": zero_test_metrics_mw,
                    },
                    "finetuned_target_test": {
                        "loss_norm": target_test_loss,
                        "metrics_norm": target_test_metrics_norm,
                        "metrics_mw": target_test_metrics_mw,
                    },
                    "improvement_vs_zero_shot": improvement,
                },
                f,
                indent=2,
            )

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(metrics_path))
        print(f"\nFine-tuned model saved to {model_path}")
        print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
