import numpy as np

from src.benchmarking.common import (
    build_prediction_intervals,
    compute_conformal_quantiles,
    compute_interval_metrics,
    denormalize_targets,
)


def test_conformal_quantiles_are_per_horizon():
    y_true = np.arange(72, dtype=np.float32).reshape(3, 24)
    y_pred = y_true - 0.5

    quantiles = compute_conformal_quantiles(y_true, y_pred, alpha=0.05)

    assert quantiles.shape == (24,)
    np.testing.assert_allclose(quantiles, np.full(24, 0.5))


def test_interval_metrics_report_perfect_coverage_and_positive_width():
    y_true = np.array([[10.0, 20.0], [11.0, 21.0]], dtype=np.float32)
    lower = y_true - 2.0
    upper = y_true + 2.0

    metrics = compute_interval_metrics(y_true, lower, upper, alpha=0.05)

    assert metrics["coverage_95"] == 1.0
    assert metrics["mean_width"] == 4.0
    assert metrics["median_width"] == 4.0
    assert metrics["interval_score"] == 4.0


def test_build_prediction_intervals_uses_horizon_quantiles():
    pred = np.array([[10.0, 20.0], [11.0, 21.0]], dtype=np.float32)
    quantiles = np.array([1.0, 3.0], dtype=np.float32)

    lower, upper = build_prediction_intervals(pred, quantiles)

    np.testing.assert_allclose(lower, np.array([[9.0, 17.0], [10.0, 18.0]]))
    np.testing.assert_allclose(upper, np.array([[11.0, 23.0], [12.0, 24.0]]))


def test_denormalize_targets_preserves_lower_pred_upper_order():
    values = np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float32)
    params = {
        "method": "standard",
        "mean": {"y_h1": 100.0, "y_h2": 200.0},
        "std": {"y_h1": 10.0, "y_h2": 20.0},
    }

    denorm = denormalize_targets(values, params, ["y_h1", "y_h2"])

    np.testing.assert_allclose(denorm, np.array([[100.0, 220.0], [90.0, 240.0]]))
