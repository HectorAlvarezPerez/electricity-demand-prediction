"""Run daily renewables benchmarks for no-external and external feature sets."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import METRICS_DIR, ensure_artifact_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the daily renewables resource benchmark")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", type=Path, default=ROOT / "data" / "processed_renewables_daily")
    p.add_argument("--output_dir", type=Path, default=METRICS_DIR / "renewables_resource_benchmark")
    p.add_argument("--models", nargs="+", choices=["xgboost", "mlp", "graphsage"], default=["xgboost", "mlp", "graphsage"])
    p.add_argument("--feature_sets", nargs="+", choices=["no_external", "external"], default=["no_external", "external"])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--xgb_estimators", type=int, default=100)
    p.add_argument("--xgb_n_jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--torch_epochs", type=int, default=300)
    p.add_argument("--torch_patience", type=int, default=20)
    p.add_argument("--log_every", type=int, default=10)
    return p.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def run_profile(args: argparse.Namespace, model: str, feature_set: str, output: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "src" / "benchmarking" / "profile_renewables.py"),
        "--model",
        model,
        "--seed",
        str(args.seed),
        "--data_dir",
        str(args.data_dir),
        "--batch_size",
        str(args.batch_size),
        "--output",
        str(output),
        "--n_estimators",
        str(args.xgb_estimators),
        "--n_jobs",
        str(args.xgb_n_jobs),
        "--epochs",
        str(args.torch_epochs),
        "--patience",
        str(args.torch_patience),
        "--log_every",
        str(args.log_every),
    ]
    if feature_set == "external":
        cmd.append("--include_external")
    log("Launching: " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def flatten_payload(model: str, feature_set: str, payload: dict) -> dict:
    row = {
        "model_name": model,
        "feature_set": feature_set,
        "seed": payload["seed"],
        "train_time_s": payload["train_time_s"],
        "peak_rss_mb": payload["peak_rss_mb"],
        "peak_vram_mb": payload.get("peak_vram_mb"),
        "model_size_mb": payload["model_size_mb"],
        "n_parameters": payload.get("n_parameters"),
        "n_trainable_parameters": payload.get("n_trainable_parameters"),
    }
    for metrics_key in ["fit_metrics", "fit_metrics_raw"]:
        prefix = "norm" if metrics_key == "fit_metrics" else "raw"
        for split_name, metrics in payload.get(metrics_key, {}).items():
            for metric_name, value in metrics.items():
                row[f"{prefix}_{split_name}_{metric_name}"] = value
    for metrics_key in ["fit_metrics_by_target", "fit_metrics_by_target_raw"]:
        prefix = "target_norm" if metrics_key == "fit_metrics_by_target" else "target_raw"
        for split_name, targets in payload.get(metrics_key, {}).items():
            for target_name, metrics in targets.items():
                for metric_name, value in metrics.items():
                    row[f"{prefix}_{split_name}_{target_name}_{metric_name}"] = value
    for split_name, profile in payload.get("inference", {}).items():
        for metric_name, value in profile.items():
            row[f"inference_{split_name}_{metric_name}"] = value
    return row


def main() -> None:
    args = parse_args()
    ensure_artifact_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[tuple[str, str], Path] = {}
    for feature_set in args.feature_sets:
        for model in args.models:
            output = args.output_dir / f"{model}_{feature_set}_seed{args.seed}.json"
            outputs[(model, feature_set)] = output
            run_profile(args, model, feature_set, output)

    payloads = {
        f"{model}_{feature_set}": json.loads(path.read_text())
        for (model, feature_set), path in outputs.items()
    }
    rows = [
        flatten_payload(model, feature_set, payloads[f"{model}_{feature_set}"])
        for (model, feature_set) in outputs
    ]
    summary_json = args.output_dir / f"renewables_resource_benchmark_seed{args.seed}.json"
    summary_csv = args.output_dir / f"renewables_resource_benchmark_seed{args.seed}.csv"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "models": payloads}, f, indent=2, ensure_ascii=False)
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    log(f"Saved -> {summary_json}")
    log(f"Saved -> {summary_csv}")


if __name__ == "__main__":
    main()
