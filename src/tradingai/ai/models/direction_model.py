"""Cabeza de clasificacion: direccion del movimiento (long / short / neutral)."""

from __future__ import annotations

from torch import nn


class DirectionHead(nn.Module):
    def __init__(self, embedding_size: int, num_classes: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size // 2, num_classes),
        )

    def forward(self, embedding):
        return self.net(embedding)  # logits (batch, num_classes)
