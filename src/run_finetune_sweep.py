"""
Run a few-shot fine-tuning sweep for the tabular MLP and aggregate results.

Example:
  python src/run_finetune_sweep.py \
      --pretrained_model artifacts/models/mlp_tabular_long_seed42.pt \
      --fractions 0.01 0.02 0.05 0.1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import METRICS_DIR, MODELS_DIR, ensure_artifact_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep few-shot target fractions for MLP fine-tuning")
    p.add_argument(
        "--pretrained_model",
        type=Path,
        default=MODELS_DIR / "mlp_tabular_long_seed42.pt",
        help="Checkpoint produced by src/train.py",
    )
    p.add_argument("--fractions", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.10])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--sampling", choices=["head", "random"], default="head")
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--experiment_name", type=str, default="mlp_target_finetune")
    p.add_argument(
        "--summary_name",
        type=str,
        default="mlp_finetune_sweep_summary.csv",
        help="File name written under artifacts/metrics/",
    )
    return p.parse_args()


def metrics_json_path(root: Path, seed: int, fraction: float, freeze_backbone: bool) -> Path:
    suffix = f"seed{seed}_frac{fraction:.3f}".replace(".", "p")
    if freeze_backbone:
        suffix += "_headonly"
    return METRICS_DIR / f"mlp_finetune_{suffix}.json"


def run_fraction(args: argparse.Namespace, fraction: float) -> Path:
    cmd = [
        sys.executable,
        str(ROOT / "src" / "train_finetune.py"),
        "--pretrained_model",
        str(args.pretrained_model),
        "--seed",
        str(args.seed),
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--patience",
        str(args.patience),
        "--target_fraction",
        str(fraction),
        "--sampling",
        args.sampling,
        "--experiment_name",
        args.experiment_name,
    ]
    if args.freeze_backbone:
        cmd.append("--freeze_backbone")

    print(f"\n=== Running fine-tune for target_fraction={fraction:.3f} ===")
    subprocess.run(cmd, cwd=ROOT, check=True)
    return metrics_json_path(ROOT, args.seed, fraction, args.freeze_backbone)


def build_row(metrics_path: Path) -> dict:
    with open(metrics_path) as f:
        payload = json.load(f)

    zero = payload["zero_shot_target_test"]
    tuned = payload["finetuned_target_test"]
    gain = payload["improvement_vs_zero_shot"]
    return {
        "target_fraction": payload["target_fraction"],
        "target_samples": payload["target_samples"],
        "sampling": payload["sampling"],
        "freeze_backbone": payload["freeze_backbone"],
        "best_epoch": payload["best_epoch"],
        "zero_shot_mae_norm": zero["metrics_norm"]["mae"],
        "finetuned_mae_norm": tuned["metrics_norm"]["mae"],
        "delta_mae_norm": gain["mae_norm_abs"],
        "delta_mae_norm_pct": gain["mae_norm_pct"],
        "zero_shot_rmse_norm": zero["metrics_norm"]["rmse"],
        "finetuned_rmse_norm": tuned["metrics_norm"]["rmse"],
        "delta_rmse_norm": gain["rmse_norm_abs"],
        "delta_rmse_norm_pct": gain["rmse_norm_pct"],
        "zero_shot_mae_mw": zero["metrics_mw"]["mae"],
        "finetuned_mae_mw": tuned["metrics_mw"]["mae"],
        "delta_mae_mw": gain["mae_mw_abs"],
        "delta_mae_mw_pct": gain["mae_mw_pct"],
        "zero_shot_rmse_mw": zero["metrics_mw"]["rmse"],
        "finetuned_rmse_mw": tuned["metrics_mw"]["rmse"],
        "delta_rmse_mw": gain["rmse_mw_abs"],
        "delta_rmse_mw_pct": gain["rmse_mw_pct"],
        "metrics_json": str(metrics_path),
    }


def main() -> None:
    args = parse_args()
    if not args.pretrained_model.exists():
        raise FileNotFoundError(args.pretrained_model)

    rows = []
    for fraction in args.fractions:
        if not 0 < fraction <= 1:
            raise ValueError(f"Invalid fraction {fraction}. Expected values in (0, 1].")
        metrics_path = run_fraction(args, fraction)
        rows.append(build_row(metrics_path))

    summary = pd.DataFrame(rows).sort_values("target_fraction").reset_index(drop=True)
    ensure_artifact_dirs()
    summary_path = METRICS_DIR / args.summary_name
    summary.to_csv(summary_path, index=False)

    print("\n=== Few-shot fine-tuning sweep summary ===")
    print(
        summary[
            [
                "target_fraction",
                "target_samples",
                "zero_shot_mae_mw",
                "finetuned_mae_mw",
                "delta_mae_mw",
                "delta_mae_mw_pct",
            ]
        ].to_string(index=False)
    )
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
