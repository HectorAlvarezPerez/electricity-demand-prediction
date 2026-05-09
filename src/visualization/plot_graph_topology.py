"""
Generate the static GraphSAGE country graph used in the demand benchmark.

The graph is built from train-split demand correlations with the same
`build_static_graph(..., top_k=3)` helper used by the model pipeline.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.graph_dataset import build_static_graph
from src.data.preprocess import ALL_CODES, TARGET_CODE
from src.paths import FIGURES_DIR, PROCESSED_DATA_DIR, ensure_artifact_dirs

TOP_K = 3
OUT_FIG = FIGURES_DIR / "graph_topology.png"
DISPLAY_ORDER = ["DE", "FR", "GR", "IT", "NL", "PT", "ES", "BE"]


def circular_layout(nodes: list[str]) -> dict[str, tuple[float, float]]:
    ordered = [node for node in DISPLAY_ORDER if node in nodes]
    positions = {}
    for idx, node in enumerate(ordered):
        angle = math.pi / 2 - idx * (2 * math.pi / len(ordered))
        positions[node] = (math.cos(angle), math.sin(angle))
    return positions


def undirected_edges(nodes: list[str], top_k: int) -> list[tuple[str, str, float]]:
    train = pd.read_parquet(PROCESSED_DATA_DIR / "train.parquet")
    edge_index, edge_weight = build_static_graph(train, nodes=nodes, top_k=top_k)

    seen: set[tuple[str, str]] = set()
    edges = []
    for src, dst, weight in zip(
        edge_index[0].tolist(),
        edge_index[1].tolist(),
        edge_weight.tolist(),
        strict=True,
    ):
        pair = tuple(sorted((nodes[src], nodes[dst])))
        if pair in seen or pair[0] == pair[1]:
            continue
        seen.add(pair)
        edges.append((pair[0], pair[1], float(weight)))
    return edges


def main() -> None:
    ensure_artifact_dirs()
    nodes = sorted(ALL_CODES)
    positions = circular_layout(nodes)
    edges = undirected_edges(nodes, TOP_K)

    fig, ax = plt.subplots(figsize=(8, 6))
    for source, target, weight in edges:
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        ax.plot(
            [x0, x1],
            [y0, y1],
            color="#6f7782",
            linewidth=0.8 + 2.2 * abs(weight),
            alpha=0.55,
            zorder=1,
        )

    for node in nodes:
        x, y = positions[node]
        face = "#c6d9f1" if node == TARGET_CODE else "#e6e8ec"
        ax.scatter(
            [x],
            [y],
            s=2500,
            color=face,
            edgecolor="#20242a",
            linewidth=1.4,
            zorder=3,
        )
        ax.text(x, y, node, ha="center", va="center", fontsize=16, color="#111111", zorder=4)

    ax.margins(0.15)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_FIG} ({len(edges)} undirected edges)")


if __name__ == "__main__":
    main()
