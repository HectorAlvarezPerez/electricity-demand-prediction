"""Shared visual style for report figures."""
from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt


MODEL_COLORS = {
    "XGBoost": "#4E79A7",
    "MLP": "#59A14F",
    "MLP (Tabular)": "#59A14F",
    "GraphSAGE": "#C46A5A",
    "GraphSAGE (Grafs)": "#C46A5A",
    "Ridge": "#8E79A8",
    "Ridge h=24": "#8E79A8",
    "Ridge Regression": "#8E79A8",
    "Daily Naive": "#8A8A8A",
    "Naive diari": "#8A8A8A",
}

SERIES_COLORS = {
    "Demanda real": "#3A3A3A",
    "ENTSO-E dia anterior": "#D89C45",
    "MAE": "#4E79A7",
    "RMSE": "#C46A5A",
    "Validació font": "#4E79A7",
    "Validació objectiu": "#D89C45",
}

CONDITION_COLORS = {
    "demand_only": "#8A8A8A",
    "without_features": "#8A8A8A",
    "temporal_only": "#4E79A7",
    "with_features": "#4E79A7",
    "weather_only": "#59A14F",
    "all_features": "#D89C45",
}

METRIC_COLORS = {
    "target_test_mae": "#4E79A7",
    "train_time_s": "#D89C45",
    "peak_rss_mb": "#59A14F",
    "target_inf_mean_ms": "#C46A5A",
}

FEATURE_FAMILY_COLORS = {
    "demand": "#4E79A7",
    "calendar": "#59A14F",
    "weather": "#D89C45",
    "country": "#8E79A8",
    "other": "#8A8A8A",
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
    ax.grid(axis="y", alpha=grid_alpha, color="#D7D7D7", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#B8B8B8")


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
