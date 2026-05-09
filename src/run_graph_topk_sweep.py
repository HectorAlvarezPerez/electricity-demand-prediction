"""Run a GraphSAGE sensitivity sweep over graph density (top-k neighbors)."""
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

from src.benchmarking.common import aggregate_numeric_by_group
from src.paths import METRICS_DIR, ensure_artifact_dirs
from src.run_resource_benchmark import flatten_metrics


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_ts()}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GraphSAGE top-k sensitivity experiments")
    p.add_argument("--top_k_values", type=int, nargs="+", default=[1, 2, 3, 5, 7])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    p.add_argument("--pred_len", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--output_dir", type=Path, default=METRICS_DIR / "graph_topk_sweep")
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def run_profile(args: argparse.Namespace, *, top_k: int, seed: int, output: Path) -> dict:
    if args.skip_existing and output.exists():
        _log(f"[TopK] Reusing existing output: {output}")
        return json.loads(output.read_text(encoding="utf-8"))

    cmd = [
        sys.executable,
        str(ROOT / "src" / "benchmarking" / "profile_graphsage.py"),
        "--seed", str(seed),
        "--pred_len", str(args.pred_len),
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--log_every", str(args.log_every),
        "--graph_top_k", str(top_k),
        "--output", str(output),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    _log(f"[TopK] Launching: {' '.join(cmd)}")
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)
    _log(f"[TopK] Finished k={top_k}, seed={seed} in {time.perf_counter() - t0:.1f}s")
    return json.loads(output.read_text(encoding="utf-8"))


def aggregate_rows(rows: list[dict]) -> list[dict]:
    metric_keys = [
        "source_test_mae",
        "source_test_rmse",
        "target_test_mae",
        "target_test_rmse",
        "raw_target_test_mae",
        "raw_target_test_rmse",
        "raw_target_test_mape",
        "train_time_s",
        "peak_rss_mb",
        "model_size_mb",
        "inference_target_test_mean_ms",
        "inference_target_test_p95_ms",
        "inference_target_test_throughput_samples_s",
    ]
    available = [key for key in metric_keys if any(key in row for row in rows)]
    return aggregate_numeric_by_group(rows, group_keys=["graph_top_k"], metric_keys=available)


def main() -> None:
    args = parse_args()
    ensure_artifact_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _log(
        f"[TopK] Sweep start | top_k_values={args.top_k_values} | seeds={args.seeds} | "
        f"epochs={args.epochs} | patience={args.patience}"
    )

    rows: list[dict] = []
    payloads: dict[str, dict] = {}
    for top_k in args.top_k_values:
        for seed in args.seeds:
            output = args.output_dir / f"graphsage_k{top_k}_seed{seed}.json"
            payload = run_profile(args, top_k=top_k, seed=seed, output=output)
            row = flatten_metrics("graphsage", payload)
            row["graph_top_k"] = top_k
            row["graph_n_directed_edges"] = payload.get("graph", {}).get("n_directed_edges")
            row["graph_n_undirected_edges"] = payload.get("graph", {}).get("n_undirected_edges")
            rows.append(row)
            payloads[f"k{top_k}_seed{seed}"] = payload

    aggregate = aggregate_rows(rows)
    rows_csv = args.output_dir / "graph_topk_sweep_rows.csv"
    aggregate_csv = args.output_dir / "graph_topk_sweep_aggregate.csv"
    summary_json = args.output_dir / "graph_topk_sweep_summary.json"

    pd.DataFrame(rows).to_csv(rows_csv, index=False)
    pd.DataFrame(aggregate).to_csv(aggregate_csv, index=False)
    summary_json.write_text(
        json.dumps(
            {
                "top_k_values": args.top_k_values,
                "seeds": args.seeds,
                "pred_len": args.pred_len,
                "rows": rows,
                "aggregate": aggregate,
                "models": payloads,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _log(f"[TopK] Rows CSV written: {rows_csv}")
    _log(f"[TopK] Aggregate CSV written: {aggregate_csv}")
    _log(f"[TopK] Summary JSON written: {summary_json}")


if __name__ == "__main__":
    main()
