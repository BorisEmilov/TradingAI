"""Modelo multi-temporalidad y multi-tarea.

Un encoder por temporalidad (D1, H1, M15, M5) codifica su propia secuencia; los 4
embeddings se concatenan y pasan por una capa de fusion, replicando el analisis
top-down de un trader: sesgo en D1 -> estructura en H1 -> entrada en M15/M5. La
salida fusionada alimenta 4 cabezas en paralelo: direccion, entrada, TP/SL y en que
temporalidad ejecutar la entrada (M15 por defecto, M5 en casos puntuales).
"""

from __future__ import annotations

import torch
from torch import nn

from tradingai.ai.data.multi_timeframe import TIMEFRAMES
from tradingai.ai.models.architectures.lstm import LSTMEncoder
from tradingai.ai.models.architectures.transformer import TransformerEncoder
from tradingai.ai.models.direction_model import DirectionHead
from tradingai.ai.models.entry_model import EntryHead
from tradingai.ai.models.entry_timeframe_model import EntryTimeframeHead
from tradingai.ai.models.tp_sl_model import TpSlHead

_ENCODERS = {
    "transformer": TransformerEncoder,
    "lstm": LSTMEncoder,
}


class MultiTimeframeTradingModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        architecture: str = "transformer",
        hidden_size: int = 128,
        fusion_hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        timeframes: tuple[str, ...] = TIMEFRAMES,
    ) -> None:
        super().__init__()
        if architecture not in _ENCODERS:
            raise ValueError(f"Arquitectura desconocida: {architecture}. Opciones: {list(_ENCODERS)}")

        self.timeframes = timeframes
        encoder_cls = _ENCODERS[architecture]
        self.encoders = nn.ModuleDict(
            {
                tf: encoder_cls(
                    n_features=n_features,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for tf in timeframes
            }
        )

        fusion_input_size = hidden_size * len(timeframes)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.direction_head = DirectionHead(fusion_hidden_size, dropout=dropout)
        self.entry_head = EntryHead(fusion_hidden_size, dropout=dropout)
        self.tp_sl_head = TpSlHead(fusion_hidden_size, dropout=dropout)
        self.entry_timeframe_head = EntryTimeframeHead(fusion_hidden_size, dropout=dropout)

    def forward(self, sequences: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embeddings = [self.encoders[tf](sequences[tf]) for tf in self.timeframes]
        fused = self.fusion(torch.cat(embeddings, dim=-1))

        tp_mult, sl_mult = self.tp_sl_head(fused)
        return {
            "direction_logits": self.direction_head(fused),
            "entry_offset": self.entry_head(fused),
            "tp_mult": tp_mult,
            "sl_mult": sl_mult,
            "entry_timeframe_logits": self.entry_timeframe_head(fused),
        }


def build_model(config: dict, n_features: int) -> MultiTimeframeTradingModel:
    model_cfg = config["model"]
    return MultiTimeframeTradingModel(
        n_features=n_features,
        architecture=model_cfg.get("architecture", "transformer"),
        hidden_size=model_cfg.get("hidden_size", 128),
        fusion_hidden_size=model_cfg.get("fusion_hidden_size", 256),
        num_layers=model_cfg.get("num_layers", 2),
        num_heads=model_cfg.get("num_heads", 4),
        dropout=model_cfg.get("dropout", 0.1),
    )
