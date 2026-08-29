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


def confirmed_swing_lows(candles: pd.DataFrame, left: int = 3, right: int = 3, count: int = 1) -> list[float]:
    """Hasta `count` swing lows confirmados mas recientes, en orden CRONOLOGICO
    (el mas viejo primero, el mas reciente al final) -- pensado para comparar
    `resultado[-1]` (ultimo) contra `resultado[-2]` (el anterior) y detectar una
    ruptura de estructura (ver `mt5.structure_exit.structure_invalidated`)."""
    lows = candles["low"].to_numpy()
    n = len(lows)
    found: list[float] = []
    for i in range(n - right - 1, left - 1, -1):
        if lows[i] < lows[i - left:i].min() and lows[i] < lows[i + 1:i + 1 + right].min():
            found.append(float(lows[i]))
            if len(found) >= count:
                break
    found.reverse()
    return found


def confirmed_swing_highs(candles: pd.DataFrame, left: int = 3, right: int = 3, count: int = 1) -> list[float]:
    """Simetrico a `confirmed_swing_lows` para maximos."""
    highs = candles["high"].to_numpy()
    n = len(highs)
    found: list[float] = []
    for i in range(n - right - 1, left - 1, -1):
        if highs[i] > highs[i - left:i].max() and highs[i] > highs[i + 1:i + 1 + right].max():
            found.append(float(highs[i]))
            if len(found) >= count:
                break
    found.reverse()
    return found


def last_confirmed_swing_low(candles: pd.DataFrame, left: int = 3, right: int = 3) -> float | None:
    found = confirmed_swing_lows(candles, left, right, count=1)
    return found[-1] if found else None


def last_confirmed_swing_high(candles: pd.DataFrame, left: int = 3, right: int = 3) -> float | None:
    found = confirmed_swing_highs(candles, left, right, count=1)
    return found[-1] if found else None
