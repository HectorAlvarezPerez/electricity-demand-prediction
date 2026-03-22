"""
Plot per-country MAE comparison for the day-ahead benchmark.

Input:
    results/day_ahead_benchmark_metrics.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "results" / "day_ahead_benchmark_metrics.json"
FIGURES_DIR = ROOT / "docs" / "figures"

DOMAIN_LABELS = {
    "source_be": "Bèlgica",
    "source_de": "Alemanya",
    "source_fr": "França",
    "source_gr": "Grècia",
    "source_it": "Itàlia",
    "source_nl": "Països Baixos",
    "source_pt": "Portugal",
    "target_es": "Espanya (Target)",
}


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{RESULTS_PATH} not found. Run src/models/evaluate_day_ahead_benchmark.py first."
        )

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    domains = list(DOMAIN_LABELS.keys())
    labels = [DOMAIN_LABELS[d] for d in domains]
    methods = [
        ("entsoe_day_ahead", "ENTSO-E day-ahead", "#1b9e77"),
        ("daily_naive_h24", "Naive h=24", "#d95f02"),
        ("ridge_h24", "Ridge h=24", "#7570b3"),
        ("xgboost_h24", "XGBoost h=24", "#1f78b4"),
    ]

    x = np.arange(len(domains))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 5.2))
    for i, (key, label, color) in enumerate(methods):
        values = [results[d][key]["mae_mw"] for d in domains]
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, values, width=width, color=color, label=label)

    ax.set_ylabel("MAE (MW)", fontsize=12)
    ax.set_title("Comparativa day-ahead (h=24) per país", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()

    out = FIGURES_DIR / "day_ahead_benchmark_per_domain.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
