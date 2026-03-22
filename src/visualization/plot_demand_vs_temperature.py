"""
Plot demand versus temperature as small multiples, one panel per country.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed_long"
FIGURES_DIR = ROOT / "docs" / "figures"
COUNTRY_ORDER = ["ES", "BE", "DE", "FR", "GR", "IT", "NL", "PT"]
COUNTRY_LABELS = {
    "ES": "Espanya (target)",
    "BE": "Bèlgica",
    "DE": "Alemanya",
    "FR": "França",
    "GR": "Grècia",
    "IT": "Itàlia",
    "NL": "Països Baixos",
    "PT": "Portugal",
}


def load_all_splits() -> pd.DataFrame:
    parts = [pd.read_parquet(DATA_DIR / f"{split}.parquet") for split in ("train", "val", "test")]
    return pd.concat(parts, ignore_index=True)


def build_binned_curve(df: pd.DataFrame, temp_col: str, demand_col: str) -> pd.DataFrame:
    temp_min = np.floor(df[temp_col].min())
    temp_max = np.ceil(df[temp_col].max())
    bins = np.arange(temp_min, temp_max + 2, 2)

    binned = df.copy()
    binned["temp_bin"] = pd.cut(binned[temp_col], bins=bins, include_lowest=True)
    curve = (
        binned.groupby("temp_bin", observed=True)
        .agg(
            mean_temp=(temp_col, "mean"),
            mean_demand=(demand_col, "mean"),
            n=(demand_col, "size"),
        )
        .reset_index(drop=True)
    )
    return curve[curve["n"] >= 50].reset_index(drop=True)


def plot_panel(ax, df: pd.DataFrame, code: str) -> float:
    temp = df["temperature_2m"].to_numpy(dtype=np.float32)
    demand = df["demand"].to_numpy(dtype=np.float32)
    corr = float(df["demand"].corr(df["temperature_2m"]))

    curve = build_binned_curve(df, "temperature_2m", "demand")
    temp_center = float(temp.mean())
    quad_coef = np.polyfit(temp - temp_center, demand, deg=2)
    quad_poly_centered = np.poly1d(quad_coef)
    temp_grid = np.linspace(temp.min(), temp.max(), 400)
    fitted_grid = quad_poly_centered(temp_grid - temp_center)

    vertex_temp = float(temp_center - quad_coef[1] / (2 * quad_coef[0]))

    ax.scatter(
        temp,
        demand,
        s=3,
        alpha=0.025,
        color="#9ecae1",
        edgecolors="none",
    )
    ax.plot(
        curve["mean_temp"],
        curve["mean_demand"],
        color="#08519c",
        linewidth=2.0,
        marker="o",
        markersize=3,
    )
    ax.plot(
        temp_grid,
        fitted_grid,
        color="#cb181d",
        linewidth=1.8,
        linestyle="--",
    )
    ax.axvline(vertex_temp, color="#cb181d", linestyle=":", linewidth=1.0)

    title = COUNTRY_LABELS[code]
    if code == "ES":
        title = f"{title}"
        ax.set_facecolor("#fff8e7")
    ax.set_title(title, fontsize=10.5)
    ax.text(
        0.04,
        0.94,
        f"r = {corr:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#cccccc"},
    )
    ax.grid(alpha=0.15)
    ax.tick_params(labelsize=8.5)
    return corr


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_all_splits()

    fig, axes = plt.subplots(2, 4, figsize=(15, 8.5), sharex=True, sharey=True)
    corrs = {}
    for ax, code in zip(axes.flat, COUNTRY_ORDER):
        country_df = df[df["country_code"] == code].copy()
        corrs[code] = plot_panel(ax, country_df, code)

    for ax in axes[:, 0]:
        ax.set_ylabel("Demanda (MW)", fontsize=10)
    for ax in axes[-1, :]:
        ax.set_xlabel("Temperatura horària (°C)", fontsize=10)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#9ecae1",
            markeredgecolor="none",
            alpha=0.5,
            markersize=7,
            label="Observacions horàries",
        ),
        Line2D(
            [0],
            [0],
            color="#08519c",
            linewidth=2.0,
            marker="o",
            markersize=4,
            label="Mitjana per bins de 2 °C",
        ),
        Line2D(
            [0],
            [0],
            color="#cb181d",
            linewidth=1.8,
            linestyle="--",
            label="Ajust quadràtic",
        ),
    ]

    fig.suptitle("Demanda vs temperatura per país", fontsize=16, y=0.98)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=10,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.92, bottom=0.10)

    out = FIGURES_DIR / "demand_vs_temperature_es.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved -> {out}")
    for code in COUNTRY_ORDER:
        print(f"{code}: corr={corrs[code]:.4f}")


if __name__ == "__main__":
    main()
