"""Encoder Transformer para secuencias de velas (arquitectura por defecto)."""

from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, max_len: int = 2048) -> None:
        super().__init__()
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerEncoder(nn.Module):
    """Codifica (batch, seq_len, n_features) -> embedding (batch, hidden_size) via mean pooling."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        h = self.encoder(h)
        return h.mean(dim=1)  # pooling temporal -> embedding de la secuencia
