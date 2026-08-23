"""Contrato de datos entre la IA (prediccion) y el ejecutor MT5 (accion).

Toda la comunicacion entre `tradingai.ai` y `tradingai.mt5` pasa por un objeto
`TradingSignal`. Esto mantiene ambos modulos desacoplados: la IA no sabe nada
de ordenes ni de MT5, y el ejecutor no sabe nada de modelos ni de features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


# Piso duro de risk:reward para cualquier senal: la IA puede pedir mas, nunca menos.
# Compartido entre el etiquetado de entrenamiento (ai.training.dataset) y el
# RiskManager (mt5.risk_manager) para que ambos lados apliquen la misma regla.
MIN_RISK_REWARD_RATIO = 2.0


@dataclass
class TradingSignal:
    symbol: str
    timeframe: str
    timestamp: datetime

    direction: Direction
    confidence: float  # 0.0 - 1.0

    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None

    # Temporalidad de ejecucion de la entrada: "M15" (por defecto) o "M5" (casos puntuales).
    entry_timeframe: str = "M15"

    # Contexto que origino la senal (order blocks, FVGs, etc.), util para logging/auditoria.
    rationale: dict = field(default_factory=dict)

    def is_actionable(self, confidence_threshold: float) -> bool:
        return self.direction != Direction.NEUTRAL and self.confidence >= confidence_threshold

    @property
    def risk_reward_ratio(self) -> float | None:
        if self.entry_price is None or self.stop_loss is None or self.take_profit is None:
            return None
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk if risk > 0 else None
