"""
Generate a Pearson correlation heatmap of hourly demand across all countries.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import FIGURES_DIR, PROCESSED_DATA_DIR, ensure_artifact_dirs

DATA_DIR = PROCESSED_DATA_DIR

COUNTRY_ORDER = ["BE", "DE", "ES", "FR", "GR", "IT", "NL", "PT"]
COUNTRY_LABELS = {
    "BE": "Bèlgica",
    "DE": "Alemanya",
    "ES": "Espanya",
    "FR": "França",
    "GR": "Grècia",
    "IT": "Itàlia",
    "NL": "P. Baixos",
    "PT": "Portugal",
}


def load_all_splits() -> pd.DataFrame:
    frames = []
    for split in ("train", "val", "test"):
        frames.append(pd.read_parquet(DATA_DIR / f"{split}.parquet"))
    return pd.concat(frames, ignore_index=True)


def build_corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    demand = df[["utc_timestamp", "country_code", "demand"]].copy()
    pivot = (
        demand.pivot_table(
            index="utc_timestamp",
            columns="country_code",
            values="demand",
            aggfunc="mean",
        )
        .reindex(columns=COUNTRY_ORDER)
        .dropna()
        .sort_index()
    )
    return pivot.corr(method="pearson")


def plot_heatmap(corr: pd.DataFrame) -> None:
    labels = [COUNTRY_LABELS[c] for c in corr.columns]
    values = corr.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-1.0, vmax=1.0)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Correlació de Pearson de la demanda horària entre països", fontsize=13)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text_color = "white" if abs(values[i, j]) > 0.65 else "black"
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color=text_color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlació")

    fig.tight_layout()
    out = FIGURES_DIR / "demand_correlation_matrix.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def main() -> None:
    ensure_artifact_dirs()
    df = load_all_splits()
    corr = build_corr_matrix(df)
    plot_heatmap(corr)
    print("\nCorrelation matrix:")
    print(corr.round(3).to_string())


if __name__ == "__main__":
    main()
