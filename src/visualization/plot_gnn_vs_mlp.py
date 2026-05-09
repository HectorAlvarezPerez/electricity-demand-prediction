import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualization.plot_style import apply_report_bar_style, color_for_model

metrics_dir = ROOT / "artifacts" / "metrics"
fig_dir = ROOT / "artifacts" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# Load metrics
gnn_path = metrics_dir / "gnn_metrics_seed42.json"
comp_path = metrics_dir / "mlp_xgboost_comparison.json"

if gnn_path.exists() and comp_path.exists():
    gnn_data = json.loads(gnn_path.read_text())
    comp_data = json.loads(comp_path.read_text())
    
    gnn_target = gnn_data["target_test"]["metrics_norm"]["mae"]
    gnn_source = gnn_data["source_test"]["metrics_norm"]["mae"]
    
    mlp_target = comp_data["mlp"]["metrics"]["target_test_mae_norm"]
    mlp_source = comp_data["mlp"]["metrics"]["source_test_mae_norm"]
    
    xgb_target = comp_data["xgboost"]["target_es"]["mae"]
    xgb_source = comp_data["xgboost"]["source_aggregate"]["mae"]
    
    labels = ["Dominis Origen (Test)", "Espanya (Zero-Shot)"]
    xgb_scores = [xgb_source, xgb_target]
    mlp_scores = [mlp_source, mlp_target]
    gnn_scores = [gnn_source, gnn_target]
    
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(2)
    width = 0.25
    
    ax.bar(x - width, xgb_scores, width, label='XGBoost', color=color_for_model("XGBoost"))
    ax.bar(x, mlp_scores, width, label='MLP (Tabular)', color=color_for_model("MLP"))
    ax.bar(x + width, gnn_scores, width, label='GraphSAGE (Grafs)', color=color_for_model("GraphSAGE"))
    
    ax.set_ylabel('MAE Normalitzat')
    ax.set_title('Comparativa Rendiment Absolut baseline vs MLP vs GraphSAGE')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    apply_report_bar_style(ax)
    
    for i, v in enumerate(xgb_scores):
        ax.text(i - width, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(mlp_scores):
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(gnn_scores):
        ax.text(i + width, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(fig_dir / "graphsage_vs_mlp.png", dpi=300)
    print("Saved artifacts/figures/graphsage_vs_mlp.png")
else:
    print("Metrics not found!")
