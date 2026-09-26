"""NUEVO -- dataclasses propias del paquete, deliberadamente NO
`trader.signal.TradingSignal` (la clase que usa el pipeline de producción)
para no acoplar estas 2 estrategias a un esquema pensado para el sistema de
scoring. Mismo principio de aislamiento que el resto de `mtf_strategies/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class MTFSignal:
    strategy: str  # "continuation" | "reversal"
    setup: str | None  # None para continuación; "setup_a" | "setup_b" | "generic" para reversión
    symbol: str
    direction: str  # "long" | "short"
    session: str  # "london" | "new_york"
    entry: float
    sl: float
    tp1: float  # siempre a 1R
    tp2: float
    risk_reward: float
    sweep_timestamp: pd.Timestamp
    mss_timestamp: pd.Timestamp
    fvg_confirmed_at: pd.Timestamp
    generated_at: pd.Timestamp
    trace: dict[str, Any] = field(default_factory=dict)  # niveles/eventos usados, para logging/debug


@dataclass(frozen=True)
class NoSignal:
    strategy: str
    reason: str
    stage: str


@dataclass(frozen=True)
class ConflictDiscarded:
    """Estrategia 1 y 2 generaron señal sobre el mismo símbolo en la misma
    sesión -- ninguna se ejecuta, se loguea para revisión (decisión de
    diseño confirmada, sin prioridad fija)."""
    symbol: str
    session: str
    continuation_signal: MTFSignal
    reversal_signal: MTFSignal
