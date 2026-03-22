"""
Run a few-shot fine-tuning sweep for XGBoost and aggregate results.

Example:
  python src/run_finetune_sweep_xgb.py \
      --pretrained_model artifacts/models/baseline_xgb.json \
      --fractions 0.01 0.02 0.05 0.10
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
    p = argparse.ArgumentParser(description="Sweep few-shot target fractions for XGBoost fine-tuning")
    p.add_argument(
        "--pretrained_model",
        type=Path,
        default=MODELS_DIR / "baseline_xgb.json",
    )
    p.add_argument(
        "--metadata_path",
        type=Path,
        default=MODELS_DIR / "baseline_xgb_features.json",
    )
    p.add_argument("--fractions", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.10])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampling", choices=["head", "random"], default="head")
    p.add_argument("--additional_estimators", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=0.02)
    p.add_argument("--max_depth", type=int, default=4)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample_bytree", type=float, default=0.8)
    p.add_argument("--eval_metric", type=str, default="rmse")
    p.add_argument("--verbose", type=int, default=10)
    p.add_argument("--summary_name", type=str, default="xgb_finetune_sweep_summary.csv")
    return p.parse_args()


def metrics_json_path(root: Path, seed: int, fraction: float) -> Path:
    suffix = f"seed{seed}_frac{fraction:.3f}".replace(".", "p")
    return METRICS_DIR / f"xgb_finetune_{suffix}.json"


def run_fraction(args: argparse.Namespace, fraction: float) -> Path:
    cmd = [
        sys.executable,
        str(ROOT / "src" / "train_finetune_xgb.py"),
        "--pretrained_model",
        str(args.pretrained_model),
        "--metadata_path",
        str(args.metadata_path),
        "--seed",
        str(args.seed),
        "--target_fraction",
        str(fraction),
        "--sampling",
        args.sampling,
        "--additional_estimators",
        str(args.additional_estimators),
        "--learning_rate",
        str(args.learning_rate),
        "--max_depth",
        str(args.max_depth),
        "--subsample",
        str(args.subsample),
        "--colsample_bytree",
        str(args.colsample_bytree),
        "--eval_metric",
        args.eval_metric,
        "--verbose",
        str(args.verbose),
    ]

    print(f"\n=== Running XGBoost fine-tune for target_fraction={fraction:.3f} ===")
    subprocess.run(cmd, cwd=ROOT, check=True)
    return metrics_json_path(ROOT, args.seed, fraction)


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
        "additional_estimators": payload["additional_estimators"],
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
    if not args.metadata_path.exists():
        raise FileNotFoundError(args.metadata_path)

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

    print("\n=== XGBoost few-shot fine-tuning sweep summary ===")
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
