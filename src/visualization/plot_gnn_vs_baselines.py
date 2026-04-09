import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
metrics_dir = ROOT / "artifacts" / "metrics"
fig_dir = ROOT / "artifacts" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# Load metrics
mlp_path = metrics_dir / "mlp_metrics_seed42.json"
gnn_path = metrics_dir / "gnn_metrics_seed42.json"
xgb_path = metrics_dir / "baseline_metrics.json"

mlp_target = json.loads(mlp_path.read_text())["target_test"]["metrics_mw"]["mae"] if mlp_path.exists() else 0
gnn_target = json.loads(gnn_path.read_text())["target_test"]["metrics_mw"]["mae"] if gnn_path.exists() else 0

xgb_target = 0
if xgb_path.exists():
    xgb_data = json.loads(xgb_path.read_text())
    xgb_target = xgb_data["XGBoost"]["metrics_target"]["mae"]

mlp_source = json.loads(mlp_path.read_text())["source_test"]["metrics_mw"]["mae"] if mlp_path.exists() else 0
gnn_source = json.loads(gnn_path.read_text())["source_test"]["metrics_mw"]["mae"] if gnn_path.exists() else 0


labels = ["Source (Test)", "Target/ES (Zero-Shot)"]
gnn_scores = [gnn_source, gnn_target]
mlp_scores = [mlp_source, mlp_target]

# xgb might missing source score aggregated easily depending on JSON format so we just plot zero-shot
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(2)
width = 0.35

ax.bar(x - width/2, mlp_scores, width, label='MLP')
ax.bar(x + width/2, gnn_scores, width, label='GraphSAGE')

ax.set_ylabel('MAE (MW)')
ax.set_title('Comparativa de Rendimiento Tabular MLP vs GraphSAGE')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.tight_layout()
fig.savefig(fig_dir / "graphsage_vs_mlp.png", dpi=300)
print("Saved artifacts/figures/graphsage_vs_mlp.png")
