from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import resource
import time
from typing import Any, Callable, Iterable

import numpy as np
import torch


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def model_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024.0 * 1024.0)


def current_rss_mb() -> float:
    # On Linux ru_maxrss is reported in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def cuda_peak_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    diff = y_true - y_pred
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    mask = np.abs(y_true) > 1e-3
    mape = float(np.mean(np.abs(diff[mask] / y_true[mask])) * 100) if mask.any() else float("nan")
    return {"mae": mae, "rmse": rmse, "mape": mape}


def summarise_latencies(latencies_s: list[float], n_samples: int) -> dict[str, float]:
    if not latencies_s:
        return {
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "total_s": 0.0,
            "throughput_samples_s": 0.0,
        }

    lat = np.asarray(latencies_s, dtype=np.float64)
    total_s = float(lat.sum())
    return {
        "mean_ms": float(lat.mean() * 1000.0),
        "median_ms": float(np.median(lat) * 1000.0),
        "p95_ms": float(np.percentile(lat, 95) * 1000.0),
        "total_s": total_s,
        "throughput_samples_s": float(n_samples / total_s) if total_s > 0 else 0.0,
    }


def chunk_indices(n_items: int, batch_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, n_items, batch_size):
        yield start, min(start + batch_size, n_items)


def profile_numpy_batches(
    X: np.ndarray,
    predict_batch: Callable[[np.ndarray], np.ndarray],
    *,
    batch_size: int,
    warmup_batches: int = 3,
    timed_batches: int = 20,
) -> dict[str, float]:
    n = len(X)
    if n == 0:
        return summarise_latencies([], 0)

    slices = list(chunk_indices(n, batch_size))
    if not slices:
        return summarise_latencies([], 0)

    # Warm-up.
    for start, end in slices[:warmup_batches]:
        _ = predict_batch(X[start:end])

    latencies: list[float] = []
    timed = slices[warmup_batches : warmup_batches + timed_batches]
    if not timed:
        timed = slices[warmup_batches:] or slices[:1]

    for start, end in timed:
        t0 = time.perf_counter()
        _ = predict_batch(X[start:end])
        latencies.append(time.perf_counter() - t0)

    n_samples = sum(end - start for start, end in timed)
    result = summarise_latencies(latencies, n_samples)
    result.update({"n_batches": float(len(timed)), "n_samples": float(n_samples), "batch_size": float(batch_size)})
    return result


def profile_torch_batches(
    loader,
    infer_batch: Callable[[Any], torch.Tensor],
    *,
    warmup_batches: int = 3,
    timed_batches: int = 20,
) -> dict[str, float]:
    it = iter(loader)

    warmup_seen = 0
    while warmup_seen < warmup_batches:
        try:
            batch = next(it)
        except StopIteration:
            break
        _ = infer_batch(batch)
        warmup_seen += 1

    latencies: list[float] = []
    n_samples = 0
    timed_seen = 0
    while timed_seen < timed_batches:
        try:
            batch = next(it)
        except StopIteration:
            break
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = infer_batch(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)
        n_samples += int(pred.shape[0])
        timed_seen += 1

    result = summarise_latencies(latencies, n_samples)
    result.update({"n_batches": float(timed_seen), "n_samples": float(n_samples)})
    return result


@dataclass
class ModelBenchmarkOutput:
    model_name: str
    seed: int
    feature_cols: list[str]
    target_cols: list[str]
    n_parameters: int | None
    n_trainable_parameters: int | None
    model_size_mb: float
    peak_rss_mb: float
    peak_vram_mb: float | None
    train_time_s: float
    fit_metrics: dict[str, dict[str, float]]
    inference: dict[str, dict[str, float]]
    artifact_paths: dict[str, str]

    def to_dict(self) -> dict:
        payload = asdict(self)
        return payload
