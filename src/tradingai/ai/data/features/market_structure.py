"""Estructura de mercado: swing highs/lows y rupturas de estructura (BOS/CHoCH).

BOS  (Break of Structure)   -> continuacion de tendencia (nuevo swing en la misma direccion)
CHoCH (Change of Character) -> posible cambio de tendencia (rompe el swing contrario)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_swings(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Marca swing highs (1) / swing lows (-1) comparando con `lookback` velas a cada lado."""
    out = df.copy()
    highs, lows = out["high"].to_numpy(), out["low"].to_numpy()
    n = len(out)

    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        window_high = highs[i - lookback : i + lookback + 1]
        window_low = lows[i - lookback : i + lookback + 1]
        if highs[i] == window_high.max():
            swing_high[i] = True
        if lows[i] == window_low.min():
            swing_low[i] = True

    out["swing_high"] = swing_high
    out["swing_low"] = swing_low
    return out


def detect_structure_breaks(df: pd.DataFrame) -> pd.DataFrame:
    """Etiqueta BOS/CHoCH comparando el cierre contra el ultimo swing relevante.

    Requiere que `detect_swings` se haya ejecutado antes (columnas swing_high/swing_low).
    """
    out = df.copy()
    if "swing_high" not in out.columns or "swing_low" not in out.columns:
        raise ValueError("Ejecuta detect_swings() antes de detect_structure_breaks().")

    last_swing_high = np.nan
    last_swing_low = np.nan
    trend = 0  # 1 = alcista, -1 = bajista, 0 = indefinido

    events = []
    for i in range(len(out)):
        close = out["close"].iloc[i]
        event = None

        if not np.isnan(last_swing_high) and close > last_swing_high:
            event = "BOS_up" if trend >= 0 else "CHoCH_up"
            trend = 1
        elif not np.isnan(last_swing_low) and close < last_swing_low:
            event = "BOS_down" if trend <= 0 else "CHoCH_down"
            trend = -1

        if out["swing_high"].iloc[i]:
            last_swing_high = out["high"].iloc[i]
        if out["swing_low"].iloc[i]:
            last_swing_low = out["low"].iloc[i]

        events.append(event)

    out["structure_event"] = events
    out["trend"] = pd.Series(events).apply(
        lambda e: 1 if e in ("BOS_up", "CHoCH_up") else (-1 if e in ("BOS_down", "CHoCH_down") else np.nan)
    ).ffill().fillna(0).to_numpy()
    return out
