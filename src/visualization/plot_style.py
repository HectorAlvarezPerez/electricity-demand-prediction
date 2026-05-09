"""Shared visual style for report figures."""
from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt


MODEL_COLORS = {
    "XGBoost": "#1f78b4",
    "MLP": "#33a02c",
    "MLP (Tabular)": "#33a02c",
    "GraphSAGE": "#e31a1c",
    "GraphSAGE (Grafs)": "#e31a1c",
    "Ridge": "#6a3d9a",
    "Ridge h=24": "#6a3d9a",
    "Ridge Regression": "#6a3d9a",
    "Daily Naive": "#7f7f7f",
    "Naive diari": "#7f7f7f",
}

SERIES_COLORS = {
    "Demanda real": "#222222",
    "ENTSO-E dia anterior": "#ff7f00",
    "MAE": "#1f78b4",
    "RMSE": "#e31a1c",
    "Validació font": "#1f78b4",
    "Validació objectiu": "#ff7f00",
}

CONDITION_COLORS = {
    "demand_only": "#7f7f7f",
    "without_features": "#7f7f7f",
    "temporal_only": "#1f78b4",
    "with_features": "#1f78b4",
    "weather_only": "#33a02c",
    "all_features": "#ff7f00",
}

METRIC_COLORS = {
    "target_test_mae": "#1f78b4",
    "train_time_s": "#ff7f00",
    "peak_rss_mb": "#33a02c",
    "target_inf_mean_ms": "#e31a1c",
}

FEATURE_FAMILY_COLORS = {
    "demand": "#1f78b4",
    "calendar": "#33a02c",
    "weather": "#ff7f00",
    "country": "#6a3d9a",
    "other": "#7f7f7f",
}


def color_for_model(label: str) -> str:
    """Return the canonical color for a model-like label."""
    clean = label.strip()
    return MODEL_COLORS.get(clean, MODEL_COLORS.get(clean.split(" h=")[0], "#7f7f7f"))


def color_for_condition(key: str) -> str:
    return CONDITION_COLORS[key]


def color_for_series(label: str) -> str:
    return SERIES_COLORS.get(label, "#7f7f7f")


def color_for_metric(key: str) -> str:
    return METRIC_COLORS.get(key, "#7f7f7f")


def apply_report_bar_style(ax, *, grid_alpha: float = 0.25) -> None:
    ax.grid(axis="y", alpha=grid_alpha)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def annotate_vertical_bars(
    ax,
    bars: Iterable,
    *,
    fmt: str = "{:.3f}",
    padding: float | None = None,
    fontsize: int = 9,
    fontweight: str | None = None,
) -> None:
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    text_pad = padding if padding is not None else y_range * 0.012
    bar_list = list(bars)
    if ax.get_yscale() == "linear" and bar_list:
        heights = [bar.get_height() for bar in bar_list]
        y_min = min(y_min, min(0.0, min(heights)) - text_pad * 4)
        y_max = max(y_max, max(0.0, max(heights)) + text_pad * 4)
        ax.set_ylim(y_min, y_max)
    for bar in bar_list:
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        offset = text_pad if height >= 0 else -text_pad
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va=va,
            fontsize=fontsize,
            fontweight=fontweight,
        )


def rotate_xticks(ax, *, rotation: int = 25, fontsize: int | None = None) -> None:
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_ha("right")
        if fontsize is not None:
            label.set_fontsize(fontsize)


def set_default_figure_style() -> None:
    plt.rcParams.update(
        {
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )
