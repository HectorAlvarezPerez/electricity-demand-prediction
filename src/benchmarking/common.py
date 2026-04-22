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
import pandas as pd
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


def _as_horizon_array(values: Any, target_cols: list[str] | None = None) -> np.ndarray:
    if hasattr(values, "reindex") and target_cols is not None:
        values = values.reindex(target_cols)
    elif isinstance(values, dict) and target_cols is not None:
        values = [values[col] for col in target_cols]
    elif hasattr(values, "to_numpy"):
        values = values.to_numpy()
    return np.asarray(values, dtype=np.float32)


def denormalize_targets(values: np.ndarray, target_params: dict, target_cols: list[str] | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    method = target_params.get("method", "standard")
    if method == "standard":
        mean = _as_horizon_array(target_params["mean"], target_cols)
        std = _as_horizon_array(target_params["std"], target_cols)
        std = np.where(std == 0, 1.0, std)
        return values * std + mean
    if method == "minmax":
        min_values = _as_horizon_array(target_params["min"], target_cols)
        max_values = _as_horizon_array(target_params["max"], target_cols)
        return values * (max_values - min_values) + min_values
    raise ValueError(f"Unknown normalisation method: {method!r}")


def compute_conformal_quantiles(y_true: np.ndarray, y_pred: np.ndarray, *, alpha: float = 0.05) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}")
    if y_true.ndim != 2:
        raise ValueError(f"Expected 2D arrays shaped (n_samples, n_horizons), got {y_true.ndim}D")
    if len(y_true) == 0:
        raise ValueError("Cannot calibrate conformal intervals with an empty calibration set")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    residuals = np.abs(y_true - y_pred)
    n = residuals.shape[0]
    quantile_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(residuals, quantile_level, axis=0, method="higher").astype(np.float32)


def build_prediction_intervals(y_pred: np.ndarray, quantiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_pred = np.asarray(y_pred, dtype=np.float32)
    quantiles = np.asarray(quantiles, dtype=np.float32)
    if y_pred.ndim != 2:
        raise ValueError(f"Expected y_pred to be 2D, got {y_pred.ndim}D")
    if quantiles.shape != (y_pred.shape[1],):
        raise ValueError(f"Expected quantiles shape {(y_pred.shape[1],)}, got {quantiles.shape}")
    return y_pred - quantiles, y_pred + quantiles


def compute_interval_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32)
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    if y_true.shape != lower.shape or y_true.shape != upper.shape:
        raise ValueError("y_true, lower and upper must have the same shape")

    inside = (y_true >= lower) & (y_true <= upper)
    width = upper - lower
    below = np.maximum(lower - y_true, 0.0)
    above = np.maximum(y_true - upper, 0.0)
    interval_score = width + (2.0 / alpha) * below + (2.0 / alpha) * above
    level_pct = int(round((1.0 - alpha) * 100))
    return {
        f"coverage_{level_pct}": float(np.mean(inside)),
        "mean_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
        "interval_score": float(np.mean(interval_score)),
    }


def build_conformal_interval_report(
    *,
    calibrations: dict[str, tuple[np.ndarray, np.ndarray]],
    evaluations: dict[str, tuple[np.ndarray, np.ndarray]],
    target_params: dict,
    target_cols: list[str],
    alpha: float = 0.05,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "level": float(1.0 - alpha),
        "alpha": float(alpha),
        "calibrations": {},
        "metrics": {},
    }
    for calibration_name, (calib_true, calib_pred) in calibrations.items():
        quantiles_norm = compute_conformal_quantiles(calib_true, calib_pred, alpha=alpha)
        zeros = np.zeros_like(quantiles_norm, dtype=np.float32).reshape(1, -1)
        quantiles_mw = denormalize_targets(quantiles_norm.reshape(1, -1), target_params, target_cols) - denormalize_targets(
            zeros, target_params, target_cols
        )
        report["calibrations"][calibration_name] = {
            "n_samples": int(np.asarray(calib_true).shape[0]),
            "quantiles_norm": quantiles_norm.tolist(),
            "quantiles_mw": quantiles_mw.reshape(-1).tolist(),
        }
        report["metrics"][calibration_name] = {}
        for split_name, (eval_true, eval_pred) in evaluations.items():
            lower_norm, upper_norm = build_prediction_intervals(eval_pred, quantiles_norm)
            eval_true_mw = denormalize_targets(eval_true, target_params, target_cols)
            lower_mw = denormalize_targets(lower_norm, target_params, target_cols)
            upper_mw = denormalize_targets(upper_norm, target_params, target_cols)
            report["metrics"][calibration_name][split_name] = {
                "norm": compute_interval_metrics(eval_true, lower_norm, upper_norm, alpha=alpha),
                "mw": compute_interval_metrics(eval_true_mw, lower_mw, upper_mw, alpha=alpha),
            }
    return report


def build_prediction_interval_frame(
    *,
    model_name: str,
    split_name: str,
    calibration_name: str,
    metadata: pd.DataFrame,
    y_true_norm: np.ndarray,
    y_pred_norm: np.ndarray,
    quantiles_norm: np.ndarray,
    target_params: dict,
    target_cols: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    lower_norm, upper_norm = build_prediction_intervals(y_pred_norm, quantiles_norm)
    y_true_mw = denormalize_targets(y_true_norm, target_params, target_cols)
    y_pred_mw = denormalize_targets(y_pred_norm, target_params, target_cols)
    lower_mw = denormalize_targets(lower_norm, target_params, target_cols)
    upper_mw = denormalize_targets(upper_norm, target_params, target_cols)

    meta = metadata[["utc_timestamp", "country_code"]].reset_index(drop=True).copy()
    if len(meta) != y_true_mw.shape[0]:
        raise ValueError(f"Metadata rows ({len(meta)}) do not match prediction rows ({y_true_mw.shape[0]})")

    level_pct = int(round((1.0 - alpha) * 100))
    frames: list[pd.DataFrame] = []
    for horizon_idx, target_col in enumerate(target_cols):
        horizon = horizon_idx + 1
        frame = meta.copy()
        frame["model_name"] = model_name
        frame["split"] = split_name
        frame["calibration"] = calibration_name
        frame["horizon"] = horizon
        frame["target_col"] = target_col
        frame["forecast_timestamp"] = pd.to_datetime(frame["utc_timestamp"], utc=True) + pd.to_timedelta(horizon, unit="h")
        frame["y_true_mw"] = y_true_mw[:, horizon_idx]
        frame["pred_mw"] = y_pred_mw[:, horizon_idx]
        frame[f"lower_{level_pct}_mw"] = lower_mw[:, horizon_idx]
        frame[f"upper_{level_pct}_mw"] = upper_mw[:, horizon_idx]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


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
