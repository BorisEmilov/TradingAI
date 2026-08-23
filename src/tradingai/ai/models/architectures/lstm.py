"""Encoder LSTM como alternativa mas ligera al Transformer."""

from __future__ import annotations

import torch
from torch import nn


class LSTMEncoder(nn.Module):
    """Codifica (batch, seq_len, n_features) -> embedding (batch, hidden_size) via ultimo estado oculto."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        **_ignored,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.output_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]  # ultima capa, ultimo paso temporal
