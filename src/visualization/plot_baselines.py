"""
Reads results/baseline_metrics.json (produced by src/models/baselines.py)
and generates comparison bar charts saved to docs/figures/.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "results" / "baseline_metrics.json"
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


def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def plot_target_comparison(results):
    """Bar chart comparing all models on the Target domain (Zero-Shot ES)."""
    models = list(results.keys())
    mae = [results[m]["target_es"]["mae"] for m in models]
    rmse = [results[m]["target_es"]["rmse"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_mae = ax.bar(x - width / 2, mae, width, label="MAE", color="steelblue")
    bars_rmse = ax.bar(x + width / 2, rmse, width, label="RMSE", color="darkorange")

    ax.set_ylabel("Error Normalitzat", fontsize=12)
    ax.set_title(
        "Errors Zero-Shot en el Target (Espanya) per Model Base",
        fontsize=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=11)

    for bars in (bars_mae, bars_rmse):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    fig.tight_layout()
    out = FIGURES_DIR / "baselines_comparison.png"
    fig.savefig(out, dpi=300)
    print(f"Saved → {out}")
    plt.close(fig)


def plot_per_domain(results):
    """Grouped bar chart showing MAE per domain for every model."""
    models = list(results.keys())
    domains = list(DOMAIN_LABELS.keys())
    labels = [DOMAIN_LABELS[d] for d in domains]

    x = np.arange(len(domains))
    width = 0.8 / len(models)
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, model in enumerate(models):
        mae_vals = [results[model][d]["mae"] for d in domains]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, mae_vals, width, label=model, color=colors[i])

    ax.set_ylabel("MAE (Normalitzat)", fontsize=12)
    ax.set_title("MAE per País i Model Base", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=25, ha="right")
    ax.legend(fontsize=10)
    fig.tight_layout()

    out = FIGURES_DIR / "baselines_per_domain.png"
    fig.savefig(out, dpi=300)
    print(f"Saved → {out}")
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not RESULTS_PATH.exists():
        print(f"ERROR: {RESULTS_PATH} not found. Run src/models/baselines.py first.")
        return

    results = load_results()
    print(f"Loaded metrics for models: {list(results.keys())}")

    plot_target_comparison(results)
    plot_per_domain(results)

    print("\nAll plots generated successfully.")


if __name__ == "__main__":
    main()
