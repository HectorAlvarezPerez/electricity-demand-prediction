"""Run hourly renewables benchmarks for no-external and external feature sets."""
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

from src.benchmarking.common import aggregate_numeric_by_group, bootstrap_mae_ci
from src.paths import METRICS_DIR, ensure_artifact_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the hourly renewables resource benchmark")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="Run and aggregate multiple seeds")
    p.add_argument("--data_dir", type=Path, default=ROOT / "data" / "processed_renewables_hourly")
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


def run_profile(args: argparse.Namespace, model: str, feature_set: str, seed: int, output: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "src" / "benchmarking" / "profile_renewables.py"),
        "--model",
        model,
        "--seed",
        str(seed),
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


def aggregate_seed_rows(rows: list[dict]) -> list[dict]:
    metric_keys = [
        "raw_target_test_mae",
        "raw_target_test_rmse",
        "raw_target_test_mape",
        "target_raw_target_test_y_solar_mwh_mae",
        "target_raw_target_test_y_wind_mwh_mae",
        "target_raw_target_test_y_hydro_mwh_mae",
        "target_raw_target_test_y_renewable_total_mwh_mae",
        "train_time_s",
        "peak_rss_mb",
        "model_size_mb",
        "inference_target_test_mean_ms",
        "inference_target_test_p95_ms",
        "inference_target_test_throughput_samples_s",
    ]
    available = [key for key in metric_keys if any(key in row for row in rows)]
    return aggregate_numeric_by_group(rows, group_keys=["model_name", "feature_set"], metric_keys=available)


def bootstrap_target_mae(output_dir: Path, seeds: list[int], models: list[str], feature_sets: list[str]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for feature_set in feature_sets:
        for model in models:
            frames = []
            for seed in seeds:
                path = output_dir / f"{model}_{feature_set}_seed{seed}_target_test_predictions.parquet"
                if path.exists():
                    frames.append(pd.read_parquet(path, columns=["target", "y_true", "pred"]))
            if not frames:
                continue
            data = pd.concat(frames, ignore_index=True)
            key = f"{model}_{feature_set}"
            result[key] = {
                "overall": bootstrap_mae_ci(
                    data["y_true"].to_numpy(),
                    data["pred"].to_numpy(),
                    n_bootstrap=1000,
                    seed=12345,
                )
            }
            for target_name, target_df in data.groupby("target", sort=False):
                result[key][target_name] = bootstrap_mae_ci(
                    target_df["y_true"].to_numpy(),
                    target_df["pred"].to_numpy(),
                    n_bootstrap=1000,
                    seed=12345,
                )
    return result


def run_seed(args: argparse.Namespace, seed: int) -> dict[str, dict]:
    outputs: dict[tuple[str, str], Path] = {}
    for feature_set in args.feature_sets:
        for model in args.models:
            output = args.output_dir / f"{model}_{feature_set}_seed{seed}.json"
            outputs[(model, feature_set)] = output
            run_profile(args, model, feature_set, seed, output)

    payloads = {
        f"{model}_{feature_set}": json.loads(path.read_text())
        for (model, feature_set), path in outputs.items()
    }
    rows = [
        flatten_payload(model, feature_set, payloads[f"{model}_{feature_set}"])
        for (model, feature_set) in outputs
    ]
    summary_json = args.output_dir / f"renewables_resource_benchmark_seed{seed}.json"
    summary_csv = args.output_dir / f"renewables_resource_benchmark_seed{seed}.csv"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "models": payloads}, f, indent=2, ensure_ascii=False)
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    log(f"Saved -> {summary_json}")
    log(f"Saved -> {summary_csv}")
    return payloads


def main() -> None:
    args = parse_args()
    seeds = args.seeds or [args.seed]
    ensure_artifact_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_payloads: dict[str, dict[str, dict]] = {}
    all_rows: list[dict] = []
    for seed in seeds:
        log(f"Running renewables benchmark seed={seed}")
        payloads = run_seed(args, seed)
        all_payloads[str(seed)] = payloads
        for payload in payloads.values():
            all_rows.append(flatten_payload(payload["model_name"], payload["feature_set"], payload))

    if len(seeds) > 1:
        aggregate_rows = aggregate_seed_rows(all_rows)
        bootstrap = bootstrap_target_mae(args.output_dir, seeds, args.models, args.feature_sets)
        summary_json = args.output_dir / "renewables_resource_benchmark_multiseed.json"
        summary_csv = args.output_dir / "renewables_resource_benchmark_multiseed.csv"
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "seeds": seeds,
                    "rows": all_rows,
                    "aggregate": aggregate_rows,
                    "bootstrap_target_mae": bootstrap,
                    "models": all_payloads,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        pd.DataFrame(aggregate_rows).to_csv(summary_csv, index=False)
        log(f"Saved -> {summary_json}")
        log(f"Saved -> {summary_csv}")


if __name__ == "__main__":
    main()
