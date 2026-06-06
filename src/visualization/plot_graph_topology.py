"""
Generate the directed k=3 country graph used in the demand benchmark.

Each country is linked to its top-k (=3) most demand-correlated neighbours in the
training split, matching the construction rule described in the report
("k=3 veins per a cada domini"). The model itself (`build_static_graph`)
symmetrises these edges for message passing; this figure shows the directed
selection rule, which is what the report figure illustrates.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import ALL_CODES, TARGET_CODE
from src.paths import FIGURES_DIR, PROCESSED_DATA_DIR, ensure_artifact_dirs

TOP_K = 3
OUT_FIG = FIGURES_DIR / "graph_topology.png"
# ES on the left, same visual order as the report figure.
DISPLAY_ORDER = ["DE", "FR", "GR", "IT", "NL", "PT", "ES", "BE"]


def circular_layout(nodes: list[str]) -> dict[str, tuple[float, float]]:
    ordered = [node for node in DISPLAY_ORDER if node in nodes]
    positions = {}
    for idx, node in enumerate(ordered):
        angle = math.pi / 2 - idx * (2 * math.pi / len(ordered))
        positions[node] = (math.cos(angle), math.sin(angle))
    return positions


def directed_topk_edges(nodes: list[str], top_k: int) -> list[tuple[str, str, float]]:
    """For each node, its top_k most demand-correlated neighbours in train."""
    train = pd.read_parquet(PROCESSED_DATA_DIR / "train.parquet")
    corr = train.pivot(index="utc_timestamp", columns="country_code", values="demand")[nodes].corr()
    edges: list[tuple[str, str, float]] = []
    for src in nodes:
        row = corr.loc[src].drop(labels=[src])
        for dst in row.sort_values(ascending=False).head(top_k).index:
            edges.append((src, dst, float(corr.loc[src, dst])))
    return edges


def main() -> None:
    ensure_artifact_dirs()
    nodes = sorted(ALL_CODES)
    positions = circular_layout(nodes)
    edges = directed_topk_edges(nodes, TOP_K)
    weights = np.array([w for *_, w in edges])
    wmin, wmax = weights.min(), weights.max()

    fig, ax = plt.subplots(figsize=(8, 6.8))
    # Curved directed arrows so reciprocal A->B / B->A pairs stay distinguishable.
    for source, target, weight in edges:
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        norm = (weight - wmin) / (wmax - wmin) if wmax > wmin else 0.5
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                connectionstyle="arc3,rad=0.13",
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=1.0 + 2.6 * norm,
                color=plt.cm.Blues(0.35 + 0.55 * norm),
                alpha=0.9,
                shrinkA=20,
                shrinkB=20,
                zorder=1,
            )
        )

    for node in nodes:
        x, y = positions[node]
        face = "#c6d9f1" if node == TARGET_CODE else "#e6e8ec"
        ax.scatter([x], [y], s=2600, color=face, edgecolor="#20242a", linewidth=1.4, zorder=3)
        ax.text(x, y, node, ha="center", va="center", fontsize=16, color="#111111", zorder=4)

    legend_handles = [
        Line2D([0], [0], color=plt.cm.Blues(0.9), lw=3.4, label="correlació alta"),
        Line2D([0], [0], color=plt.cm.Blues(0.45), lw=1.4, label="correlació baixa"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#c6d9f1",
               markeredgecolor="#20242a", markersize=12, label=f"{TARGET_CODE} (domini objectiu)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8.5, frameon=False,
              bbox_to_anchor=(1.18, 1.05))

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.8, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_FIG} ({len(edges)} directed edges, k={TOP_K})")


if __name__ == "__main__":
    main()
