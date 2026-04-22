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

from src.paths import METRICS_DIR, ensure_artifact_dirs


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_ts()}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the unified resource benchmark")
    p.add_argument("--seed", type=int, default=42)
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


def main() -> None:
    args = parse_args()
    run_t0 = time.perf_counter()
    _log(
        f"[Runner] Unified benchmark start | seed={args.seed} | pred_len={args.pred_len} | "
        f"xgb_n_jobs={args.xgb_n_jobs} | xgb_estimators={args.xgb_estimators} | "
        f"mlp_epochs={args.mlp_epochs} patience={args.mlp_patience} | "
        f"gnn_epochs={args.gnn_epochs} patience={args.gnn_patience}"
    )
    ensure_artifact_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _log(f"[Runner] Output directory: {args.output_dir}")

    outputs = {
        "xgboost": args.output_dir / f"xgb_seed{args.seed}.json",
        "mlp": args.output_dir / f"mlp_seed{args.seed}.json",
        "graphsage": args.output_dir / f"graphsage_seed{args.seed}.json",
    }
    _log("[Runner] Planned artifact outputs:")
    for key, path in outputs.items():
        _log(f"  - {key}: {path}")

    _log("[Runner] Step 1/4: XGBoost profiling")
    run_script(
        ROOT / "src" / "benchmarking" / "profile_xgb.py",
        [
            "--seed", str(args.seed),
            "--pred_len", str(args.pred_len),
            "--batch_size", str(args.xgb_batch_size),
            "--n_estimators", str(args.xgb_estimators),
            "--n_jobs", str(args.xgb_n_jobs),
            "--output", str(outputs["xgboost"]),
        ],
    )
    _log(f"[Runner] XGBoost output exists: {outputs['xgboost'].exists()}")

    _log("[Runner] Step 2/4: MLP profiling")
    run_script(
        ROOT / "src" / "benchmarking" / "profile_mlp.py",
        [
            "--seed", str(args.seed),
            "--pred_len", str(args.pred_len),
            "--batch_size", str(args.torch_batch_size),
            "--epochs", str(args.mlp_epochs),
            "--patience", str(args.mlp_patience),
            "--log_every", str(args.mlp_log_every),
            "--output", str(outputs["mlp"]),
        ],
    )
    _log(f"[Runner] MLP output exists: {outputs['mlp'].exists()}")

    _log("[Runner] Step 3/4: GraphSAGE profiling")
    run_script(
        ROOT / "src" / "benchmarking" / "profile_graphsage.py",
        [
            "--seed", str(args.seed),
            "--pred_len", str(args.pred_len),
            "--batch_size", str(args.torch_batch_size),
            "--epochs", str(args.gnn_epochs),
            "--patience", str(args.gnn_patience),
            "--log_every", str(args.gnn_log_every),
            "--output", str(outputs["graphsage"]),
        ],
    )
    _log(f"[Runner] GraphSAGE output exists: {outputs['graphsage'].exists()}")

    _log("[Runner] Aggregating per-model JSON files")
    payloads = {name: json.loads(path.read_text()) for name, path in outputs.items()}
    rows = [flatten_metrics(name, payload) for name, payload in payloads.items()]
    summary_df = pd.DataFrame(rows)

    summary_json = args.output_dir / f"resource_benchmark_seed{args.seed}.json"
    summary_csv = args.output_dir / f"resource_benchmark_seed{args.seed}.csv"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "pred_len": args.pred_len, "models": payloads}, f, indent=2, ensure_ascii=False)
    summary_df.to_csv(summary_csv, index=False)
    _log(f"[Runner] Summary JSON written: {summary_json}")
    _log(f"[Runner] Summary CSV written: {summary_csv}")

    # Generate the report-ready LaTeX fragment and comparison figure sequentially.
    _log("[Runner] Step 4/4: Exporting results to existing report section")
    run_script(
        ROOT / "src" / "visualization" / "export_resource_benchmark_report.py",
        ["--seed", str(args.seed)],
    )

    total_elapsed = time.perf_counter() - run_t0
    _log(f"[Runner] Unified benchmark completed in {total_elapsed/60.0:.1f} min")
    _log(f"Saved -> {summary_json}")
    _log(f"Saved -> {summary_csv}")


if __name__ == "__main__":
    main()
