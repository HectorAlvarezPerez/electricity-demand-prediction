"""Run the unified resource benchmark for XGBoost, MLP and GraphSAGE."""
from __future__ import annotations

import argparse
import os
import json
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


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_ts()}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the unified resource benchmark")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="Run and aggregate multiple seeds")
    p.add_argument("--pred_len", type=int, default=24)
    p.add_argument("--xgb_batch_size", type=int, default=1024)
    p.add_argument("--torch_batch_size", type=int, default=256)
    p.add_argument("--xgb_estimators", type=int, default=100)
    p.add_argument("--xgb_n_jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--mlp_epochs", type=int, default=500)
    p.add_argument("--mlp_patience", type=int, default=20)
    p.add_argument("--mlp_log_every", type=int, default=5)
    p.add_argument("--gnn_epochs", type=int, default=500)
    p.add_argument("--gnn_patience", type=int, default=20)
    p.add_argument("--gnn_log_every", type=int, default=5)
    p.add_argument("--graph_top_k", type=int, default=3)
    p.add_argument("--output_dir", type=Path, default=METRICS_DIR / "resource_benchmark")
    return p.parse_args()


def run_script(script: Path, args: list[str]) -> None:
    cmd = [sys.executable, str(script), *args]
    _log(f"[Runner] Launching: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)
    elapsed = time.perf_counter() - t0
    _log(f"[Runner] Finished: {script.name} ({elapsed:.1f}s)")


def flatten_metrics(model_name: str, payload: dict) -> dict:
    row = {
        "model_name": model_name,
        "seed": payload["seed"],
        "train_time_s": payload["train_time_s"],
        "peak_rss_mb": payload["peak_rss_mb"],
        "peak_vram_mb": payload.get("peak_vram_mb"),
        "model_size_mb": payload["model_size_mb"],
        "n_parameters": payload.get("n_parameters"),
        "n_trainable_parameters": payload.get("n_trainable_parameters"),
    }

    for split_name, metrics in payload["fit_metrics"].items():
        for metric_name, value in metrics.items():
            row[f"{split_name}_{metric_name}"] = value

    for split_name, metrics in payload.get("fit_metrics_raw", {}).items():
        for metric_name, value in metrics.items():
            row[f"raw_{split_name}_{metric_name}"] = value

    for split_name, prof in payload["inference"].items():
        for metric_name, value in prof.items():
            row[f"inference_{split_name}_{metric_name}"] = value

    for split_name, loss in payload.get("losses", {}).items():
        row[f"loss_{split_name}"] = loss

    interval_metrics = payload.get("prediction_intervals", {}).get("metrics", {})
    for calibration_name, calibration_payload in interval_metrics.items():
        for split_name, split_payload in calibration_payload.items():
            for scale_name, metrics in split_payload.items():
                for metric_name, value in metrics.items():
                    row[f"interval_{calibration_name}_{split_name}_{scale_name}_{metric_name}"] = value

    return row


def aggregate_seed_rows(rows: list[dict]) -> list[dict]:
    metric_keys = [
        "source_test_mae",
        "source_test_rmse",
        "target_test_mae",
        "target_test_rmse",
        "raw_target_test_mape",
        "train_time_s",
        "peak_rss_mb",
        "model_size_mb",
        "inference_target_test_mean_ms",
        "inference_target_test_p95_ms",
        "inference_target_test_throughput_samples_s",
        "interval_source_val_target_test_mw_coverage_95",
        "interval_source_val_target_test_mw_mean_width",
        "interval_source_val_target_test_mw_interval_score",
        "interval_target_val_target_test_mw_coverage_95",
        "interval_target_val_target_test_mw_mean_width",
        "interval_target_val_target_test_mw_interval_score",
    ]
    available = [key for key in metric_keys if any(key in row for row in rows)]
    return aggregate_numeric_by_group(rows, group_keys=["model_name"], metric_keys=available)


def bootstrap_target_mae_mw(output_dir: Path, seeds: list[int]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    model_files = {
        "xgboost": "xgb",
        "mlp": "mlp",
        "graphsage": "graphsage",
    }
    for model_name, file_prefix in model_files.items():
        frames = []
        for seed in seeds:
            path = output_dir / f"{file_prefix}_seed{seed}_prediction_intervals.parquet"
            if not path.exists():
                continue
            frame = pd.read_parquet(path, columns=["split", "calibration", "y_true_mw", "pred_mw"])
            frame = frame[frame["split"].eq("target_test")]
            if "target_val" in set(frame["calibration"].dropna().unique()):
                frame = frame[frame["calibration"].eq("target_val")]
            else:
                first_calibration = frame["calibration"].dropna().iloc[0]
                frame = frame[frame["calibration"].eq(first_calibration)]
            frames.append(frame[["y_true_mw", "pred_mw"]])
        if frames:
            data = pd.concat(frames, ignore_index=True)
            result[model_name] = bootstrap_mae_ci(
                data["y_true_mw"].to_numpy(),
                data["pred_mw"].to_numpy(),
                n_bootstrap=1000,
                seed=12345,
            )
    return result


def run_seed(args: argparse.Namespace, seed: int) -> dict[str, dict]:
    _log(
        f"[Runner] Unified benchmark seed start | seed={seed} | pred_len={args.pred_len} | "
        f"xgb_n_jobs={args.xgb_n_jobs} | xgb_estimators={args.xgb_estimators} | "
        f"mlp_epochs={args.mlp_epochs} patience={args.mlp_patience} | "
        f"gnn_epochs={args.gnn_epochs} patience={args.gnn_patience} | graph_top_k={args.graph_top_k}"
    )
    outputs = {
        "xgboost": args.output_dir / f"xgb_seed{seed}.json",
        "mlp": args.output_dir / f"mlp_seed{seed}.json",
        "graphsage": args.output_dir / f"graphsage_seed{seed}.json",
    }
    _log("[Runner] Planned artifact outputs:")
    for key, path in outputs.items():
        _log(f"  - {key}: {path}")

    _log(f"[Runner] Seed {seed} step 1/3: XGBoost profiling")
    run_script(
        ROOT / "src" / "benchmarking" / "profile_xgb.py",
        [
            "--seed", str(seed),
            "--pred_len", str(args.pred_len),
            "--batch_size", str(args.xgb_batch_size),
            "--n_estimators", str(args.xgb_estimators),
            "--n_jobs", str(args.xgb_n_jobs),
            "--output", str(outputs["xgboost"]),
        ],
    )

    _log(f"[Runner] Seed {seed} step 2/3: MLP profiling")
    run_script(
        ROOT / "src" / "benchmarking" / "profile_mlp.py",
        [
            "--seed", str(seed),
            "--pred_len", str(args.pred_len),
            "--batch_size", str(args.torch_batch_size),
            "--epochs", str(args.mlp_epochs),
            "--patience", str(args.mlp_patience),
            "--log_every", str(args.mlp_log_every),
            "--output", str(outputs["mlp"]),
        ],
    )

    _log(f"[Runner] Seed {seed} step 3/3: GraphSAGE profiling")
    run_script(
        ROOT / "src" / "benchmarking" / "profile_graphsage.py",
        [
            "--seed", str(seed),
            "--pred_len", str(args.pred_len),
            "--batch_size", str(args.torch_batch_size),
            "--epochs", str(args.gnn_epochs),
            "--patience", str(args.gnn_patience),
            "--log_every", str(args.gnn_log_every),
            "--graph_top_k", str(args.graph_top_k),
            "--output", str(outputs["graphsage"]),
        ],
    )

    payloads = {name: json.loads(path.read_text()) for name, path in outputs.items()}
    rows = [flatten_metrics(name, payload) for name, payload in payloads.items()]
    summary_df = pd.DataFrame(rows)

    summary_json = args.output_dir / f"resource_benchmark_seed{seed}.json"
    summary_csv = args.output_dir / f"resource_benchmark_seed{seed}.csv"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "pred_len": args.pred_len, "models": payloads}, f, indent=2, ensure_ascii=False)
    summary_df.to_csv(summary_csv, index=False)
    _log(f"[Runner] Seed summary JSON written: {summary_json}")
    _log(f"[Runner] Seed summary CSV written: {summary_csv}")
    return payloads


def main() -> None:
    args = parse_args()
    seeds = args.seeds or [args.seed]
    run_t0 = time.perf_counter()
    _log(
        f"[Runner] Unified benchmark start | seeds={seeds} | pred_len={args.pred_len} | "
        f"xgb_n_jobs={args.xgb_n_jobs} | xgb_estimators={args.xgb_estimators} | "
        f"mlp_epochs={args.mlp_epochs} patience={args.mlp_patience} | "
        f"gnn_epochs={args.gnn_epochs} patience={args.gnn_patience} | graph_top_k={args.graph_top_k}"
    )
    ensure_artifact_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _log(f"[Runner] Output directory: {args.output_dir}")

    all_payloads: dict[str, dict[str, dict]] = {}
    all_rows: list[dict] = []
    for seed in seeds:
        payloads = run_seed(args, seed)
        all_payloads[str(seed)] = payloads
        all_rows.extend(flatten_metrics(name, payload) for name, payload in payloads.items())

    if len(seeds) > 1:
        aggregate_rows = aggregate_seed_rows(all_rows)
        bootstrap = bootstrap_target_mae_mw(args.output_dir, seeds)
        multiseed_json = args.output_dir / "resource_benchmark_multiseed.json"
        multiseed_csv = args.output_dir / "resource_benchmark_multiseed.csv"
        with open(multiseed_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "seeds": seeds,
                    "pred_len": args.pred_len,
                    "rows": all_rows,
                    "aggregate": aggregate_rows,
                    "bootstrap_target_mae_mw": bootstrap,
                    "models": all_payloads,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        pd.DataFrame(aggregate_rows).to_csv(multiseed_csv, index=False)
        _log(f"[Runner] Multi-seed JSON written: {multiseed_json}")
        _log(f"[Runner] Multi-seed CSV written: {multiseed_csv}")

    total_elapsed = time.perf_counter() - run_t0
    _log(f"[Runner] Unified benchmark completed in {total_elapsed/60.0:.1f} min")


if __name__ == "__main__":
    main()
