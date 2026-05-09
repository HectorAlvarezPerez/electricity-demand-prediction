import pandas as pd

from src.data.graph_dataset import build_static_graph


def _toy_train_frame(nodes: list[str]) -> pd.DataFrame:
    rows = []
    for t in range(8):
        for idx, node in enumerate(nodes):
            rows.append(
                {
                    "utc_timestamp": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=t),
                    "country_code": node,
                    "demand": float((idx + 1) * (t + 1)),
                }
            )
    return pd.DataFrame(rows)


def test_top_k_is_clamped_to_complete_graph_without_self_loops():
    nodes = ["A", "B", "C", "D"]
    edge_index, edge_weight = build_static_graph(_toy_train_frame(nodes), nodes=nodes, top_k=99)

    edges = set(map(tuple, edge_index.t().tolist()))

    assert len(edges) == len(nodes) * (len(nodes) - 1)
    assert all(src != dst for src, dst in edges)
    assert edge_weight.shape[0] == len(edges)


def test_sparse_top_k_keeps_symmetric_edges():
    nodes = ["A", "B", "C", "D"]
    edge_index, _edge_weight = build_static_graph(_toy_train_frame(nodes), nodes=nodes, top_k=1)

    edges = set(map(tuple, edge_index.t().tolist()))

    assert edges
    assert all((dst, src) in edges for src, dst in edges)
    assert all(src != dst for src, dst in edges)
