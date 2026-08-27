"""Deteccion de swings (fractales) de estructura de precio.

Compartido entre `mt5` (gestion en vivo: trailing stop, extension de TP) y
`ai.evaluation` (backtest de esas mismas reglas sobre historico) -- ninguno de los
dos debe depender del otro (mismo motivo que `core.instruments`, ver 2026-08-27).

Un swing se confirma cuando una vela es mas extrema que `left` velas anteriores Y
`right` velas posteriores -- no se reacciona al ultimo minimo/maximo, que todavia
puede romperse.
"""

from __future__ import annotations

import pandas as pd


def last_confirmed_swing_low(candles: pd.DataFrame, left: int = 3, right: int = 3) -> float | None:
    lows = candles["low"].to_numpy()
    n = len(lows)
    for i in range(n - right - 1, left - 1, -1):
        if lows[i] < lows[i - left:i].min() and lows[i] < lows[i + 1:i + 1 + right].min():
            return float(lows[i])
    return None


def last_confirmed_swing_high(candles: pd.DataFrame, left: int = 3, right: int = 3) -> float | None:
    highs = candles["high"].to_numpy()
    n = len(highs)
    for i in range(n - right - 1, left - 1, -1):
        if highs[i] > highs[i - left:i].max() and highs[i] > highs[i + 1:i + 1 + right].max():
            return float(highs[i])
    return None
