"""Cabeza de regresion: take-profit y stop-loss, expresados como distancia en ATRs desde la entrada."""

from __future__ import annotations

from torch import nn


class TpSlHead(nn.Module):
    def __init__(self, embedding_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size // 2, 2),  # [tp_atr_mult, sl_atr_mult]
        )

    def forward(self, embedding):
        out = self.net(embedding)
        return out[:, 0], out[:, 1]  # tp_mult (batch,), sl_mult (batch,)
