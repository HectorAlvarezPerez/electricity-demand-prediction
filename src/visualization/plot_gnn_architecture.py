"""
Generate a clean schematic of the GraphSAGE pipeline used in the demand benchmark.

The figure is intended for the report and shows:
  - the static country graph built from train-split similarity,
  - two GraphSAGE message-passing layers,
  - the final MLP head producing the multi-step forecast.
"""
from __future__ import annotations

from math import cos, pi, sin
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import FIGURES_DIR, ensure_artifact_dirs


OUT_FIG = FIGURES_DIR / "gnn_architecture_schema.png"


def add_box(ax, xy, width, height, text, *, fc, ec, fontsize: float = 12, weight="semibold"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        linewidth=1.4,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        color="#1b1b1b",
        wrap=True,
    )


def arrow(ax, start, end, *, color="#3b3b3b", lw=1.8, style="-|>", mutation_scale=14, alpha=1.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        alpha=alpha,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(patch)


def draw_graph_panel(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.94, "Grafo estático por timestamp", ha="center", va="center", fontsize=13, weight="bold")
    ax.text(0.5, 0.88, "Nodos = países · Aristas = similitud en train", ha="center", va="center", fontsize=9.5, color="#555555")

    nodes = ["ES", "BE", "DE", "FR", "GR", "IT", "NL", "PT"]
    angles = np.linspace(pi / 2, pi / 2 + 2 * pi, len(nodes), endpoint=False)
    center = np.array([0.5, 0.46])
    radius = 0.28
    positions = {
        node: center + radius * np.array([cos(a), sin(a)])
        for node, a in zip(nodes, angles)
    }

    edges = [
        ("ES", "FR", 0.90),
        ("ES", "PT", 0.82),
        ("ES", "BE", 0.70),
        ("FR", "DE", 0.58),
        ("FR", "IT", 0.55),
        ("BE", "NL", 0.62),
        ("DE", "NL", 0.50),
        ("IT", "GR", 0.52),
    ]

    for a, b, strength in edges:
        pa = positions[a]
        pb = positions[b]
        arrow(ax, pa, pb, color="#7a7a7a", lw=2.4 * strength, mutation_scale=10, alpha=0.25 + 0.55 * strength)

    for node, pos in positions.items():
        if node == "ES":
            face = "#ffcc80"
            edge = "#ef6c00"
            radius_node = 0.045
        else:
            face = "#dbe9ff"
            edge = "#4f81bd"
            radius_node = 0.041
        circ = Circle((float(pos[0]), float(pos[1])), radius_node, facecolor=face, edgecolor=edge, linewidth=2.0, zorder=3)
        ax.add_patch(circ)
        ax.text(pos[0], pos[1], node, ha="center", va="center", fontsize=11, weight="bold", zorder=4)

    ax.text(
        0.5,
        0.10,
        "El modelo aprende del contexto de países vecinos para mejorar la predicción de ES.",
        ha="center",
        va="center",
        fontsize=9.4,
        color="#555555",
    )


def draw_pipeline_panel(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.94, "Message passing de GraphSAGE", ha="center", va="center", fontsize=13, weight="bold")

    add_box(ax, (0.06, 0.68), 0.26, 0.16, "X_t\nfeatures por nodo", fc="#eef4ff", ec="#5b8def", fontsize=11)
    add_box(ax, (0.39, 0.66), 0.22, 0.20, "GraphSAGE\ncapa 1", fc="#e8f4ea", ec="#5aa469", fontsize=12)
    add_box(ax, (0.68, 0.66), 0.22, 0.20, "GraphSAGE\ncapa 2", fc="#e8f4ea", ec="#5aa469", fontsize=12)

    arrow(ax, (0.32, 0.76), (0.39, 0.76), color="#4f81bd", lw=2.0)
    arrow(ax, (0.61, 0.76), (0.68, 0.76), color="#4f81bd", lw=2.0)

    ax.text(0.39, 0.55, "Agrega vecinos\n(mean + pesos)", ha="center", va="center", fontsize=9.6, color="#4a6f4a")
    ax.text(0.68, 0.55, "Repite el\nmessage passing", ha="center", va="center", fontsize=9.6, color="#4a6f4a")

    ax.text(0.18, 0.40, "h(0) = x", ha="center", va="center", fontsize=11)
    ax.text(0.50, 0.40, "h(1) = AGG(vecinos)", ha="center", va="center", fontsize=11)
    ax.text(0.79, 0.40, "h(2)", ha="center", va="center", fontsize=11)

    add_box(ax, (0.33, 0.12), 0.34, 0.14, "MLP head\nForecast 24 horas", fc="#fff0e8", ec="#cc6f2d", fontsize=11)
    arrow(ax, (0.79, 0.66), (0.50, 0.26), color="#cc6f2d", lw=2.0, mutation_scale=16)
    arrow(ax, (0.50, 0.66), (0.50, 0.26), color="#cc6f2d", lw=1.6, mutation_scale=14, alpha=0.75)

    ax.text(0.50, 0.03, "Salida: predicción multihorizonte para el nodo ES", ha="center", va="center", fontsize=9.4, color="#555555")


def draw_output_panel(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.94, "Interpretación de la salida", ha="center", va="center", fontsize=13, weight="bold")

    add_box(ax, (0.12, 0.70), 0.32, 0.12, "Entrada\n(pais, lags, clima)", fc="#eef4ff", ec="#5b8def", fontsize=11)
    add_box(ax, (0.56, 0.70), 0.32, 0.12, "Embeddings\npor país", fc="#e8f4ea", ec="#5aa469", fontsize=11)
    arrow(ax, (0.44, 0.76), (0.56, 0.76), color="#4f81bd", lw=2.0)

    add_box(ax, (0.20, 0.38), 0.24, 0.12, "Nodo ES\n(target)", fc="#ffcc80", ec="#ef6c00", fontsize=11)
    add_box(ax, (0.56, 0.38), 0.24, 0.12, "Vecinos\ncorrelacionados", fc="#dbe9ff", ec="#4f81bd", fontsize=11)
    arrow(ax, (0.44, 0.44), (0.56, 0.44), color="#7a7a7a", lw=1.8)

    add_box(ax, (0.23, 0.12), 0.54, 0.14, "ŷ_ES,t+1 ... ŷ_ES,t+24\nPronóstico final", fc="#fff0e8", ec="#cc6f2d", fontsize=11)
    arrow(ax, (0.70, 0.38), (0.50, 0.26), color="#cc6f2d", lw=2.0, mutation_scale=16)

    ax.text(0.50, 0.04, "La GNN no memoriza un país: aprende relaciones entre países.", ha="center", va="center", fontsize=9.4, color="#555555")


def main() -> None:
    ensure_artifact_dirs()

    fig = plt.figure(figsize=(16, 7.5), facecolor="white")
    gs = fig.add_gridspec(1, 3, wspace=0.12)

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    draw_graph_panel(axes[0])
    draw_pipeline_panel(axes[1])
    draw_output_panel(axes[2])

    fig.suptitle("Esquema de la GNN GraphSAGE para el benchmark de demanda", fontsize=16, weight="bold", y=0.98)
    fig.text(
        0.5,
        0.01,
        "Grafo estático por correlación en train · 2 capas GraphSAGE · cabeza MLP multi-horizonte",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#666666",
    )
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_FIG}")


if __name__ == "__main__":
    main()