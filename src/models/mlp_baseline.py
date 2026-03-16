"""
MLP baseline per a predicció de demanda elèctrica.

Arquitectura: flatten de la finestra d'entrada → capes FC → sortida pred_len.
"""
import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """MLP amb lags: aplana seq_len × n_features i prediu pred_len passos."""

    def __init__(
        self,
        n_features: int,
        seq_len: int = 168,
        pred_len: int = 24,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        layers: list[nn.Module] = []
        in_dim = seq_len * n_features
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, pred_len))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            (batch, pred_len)
        """
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size, -1)  # (batch, seq_len * n_features)
        return self.net(x_flat)
