"""
MLP baseline for the explicit-lag tabular forecasting dataset.
"""
import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """Tabular MLP that maps fixed lag features to a multi-step horizon."""

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

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, pred_len))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
