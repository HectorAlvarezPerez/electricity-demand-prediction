"""
GraphSAGE model for the explicit-lag tabular forecasting dataset.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class WeightedSAGEConv(MessagePassing):
    """
    Minimal GraphSAGE convolution layer that supports edge weights.
    Standard torch_geometric.nn.SAGEConv doesn't natively support edge_weights.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="mean")
        self.lin_l = nn.Linear(in_channels, out_channels)
        self.lin_r = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = self.lin_l(out)
        return out + self.lin_r(x)

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return x_j * edge_weight.view(-1, 1)


class GraphSAGEBaseline(nn.Module):
    """2-layer GraphSAGE followed by an MLP head."""

    def __init__(
        self,
        input_dim: int,
        pred_len: int = 24,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.pred_len = pred_len

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        # 2 layers of GraphSAGE
        self.conv1 = WeightedSAGEConv(input_dim, hidden_dims[0])
        self.conv2 = WeightedSAGEConv(hidden_dims[0], hidden_dims[1])
        
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], pred_len)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index, edge_weight)
        h = self.act(h)
        h = self.dropout(h)
        
        h = self.conv2(h, edge_index, edge_weight)
        h = self.act(h)
        h = self.dropout(h)
        
        return self.mlp(h)
