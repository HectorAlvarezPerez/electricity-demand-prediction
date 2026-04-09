import sys
sys.path.append("/home/hector/Escritorio/MatCAD/TFG/electricity-demand-prediction")
from src.data.graph_dataset import get_graph_dataloaders
from pathlib import Path

root = Path("/home/hector/Escritorio/MatCAD/TFG/electricity-demand-prediction")
print("Loading dataloaders...")
try:
    train_loader, val_loader, test_loader, x_cols, y_cols, f_params, t_params = get_graph_dataloaders(
        root,
        pred_len=24,
        batch_size=32,
        include_temporal=True,
        include_weather=False,
    )
    for batch in train_loader:
        print("Batch size:", batch.num_graphs)
        print("X shape:", batch.x.shape)
        print("Y shape:", batch.y.shape)
        print("Edge index shape:", batch.edge_index.shape)
        print("Edge index sample:", batch.edge_index[:, :5])
        print("Edge weight shape:", batch.edge_attr.shape)
        print("Source mask shape:", batch.source_mask.shape)
        print("Source mask sample:", batch.source_mask[:8])
        break
except Exception as e:
    import traceback
    traceback.print_exc()
