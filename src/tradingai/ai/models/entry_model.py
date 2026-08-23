"""Cabeza de regresion: precio de entrada optimo, expresado como offset % sobre el cierre actual."""

from __future__ import annotations

from torch import nn


class EntryHead(nn.Module):
    def __init__(self, embedding_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size // 2, 1),
        )

    def forward(self, embedding):
        return self.net(embedding).squeeze(-1)  # offset % (batch,)
