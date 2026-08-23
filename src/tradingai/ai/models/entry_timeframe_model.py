"""Cabeza de clasificacion binaria: en que temporalidad ejecutar la entrada.

M15 es la temporalidad de entrada por defecto; M5 se reserva para casos puntuales
donde hace falta mas precision (ver el heuristico de etiquetado en training/dataset.py).
"""

from __future__ import annotations

from torch import nn

ENTRY_TIMEFRAMES = ["M15", "M5"]


class EntryTimeframeHead(nn.Module):
    def __init__(self, embedding_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size // 2, len(ENTRY_TIMEFRAMES)),
        )

    def forward(self, embedding):
        return self.net(embedding)  # logits (batch, 2)
