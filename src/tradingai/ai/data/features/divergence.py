"""Divergencias de momentum (precio vs RSI): confirmacion clasica de agotamiento o
fuerza de tendencia, usada junto a Elliott Wave (ej. divergencia bajista en onda 5 =
señal de agotamiento del impulso).

Bajista: el precio marca un maximo mas alto que el swing anterior, pero el RSI en ese
punto es MENOR que en el swing anterior -> el impulso pierde fuerza pese al nuevo maximo.
Alcista: simetrico sobre swing lows.

Requiere que `swing_high`/`swing_low` (market_structure.py) y `rsi_14` (indicators.py)
ya esten calculados. Solo compara el swing actual contra el swing confirmado anterior
(estrictamente en el pasado) -> causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_momentum_divergence(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("swing_high", "swing_low", "rsi_14"):
        if col not in df.columns:
            raise ValueError(f"detect_momentum_divergence requiere la columna '{col}' ya calculada.")

    out = df.copy()
    n = len(out)
    bearish_div = np.zeros(n, dtype=bool)
    bullish_div = np.zeros(n, dtype=bool)

    high, low, rsi = out["high"].to_numpy(), out["low"].to_numpy(), out["rsi_14"].to_numpy()
    swing_high, swing_low = out["swing_high"].to_numpy(), out["swing_low"].to_numpy()

    last_high_price, last_high_rsi = None, None
    last_low_price, last_low_rsi = None, None

    for i in range(n):
        if swing_high[i] and not np.isnan(rsi[i]):
            if last_high_price is not None and high[i] > last_high_price and rsi[i] < last_high_rsi:
                bearish_div[i] = True
            last_high_price, last_high_rsi = high[i], rsi[i]

        if swing_low[i] and not np.isnan(rsi[i]):
            if last_low_price is not None and low[i] < last_low_price and rsi[i] > last_low_rsi:
                bullish_div[i] = True
            last_low_price, last_low_rsi = low[i], rsi[i]

    out["bearish_momentum_divergence"] = bearish_div
    out["bullish_momentum_divergence"] = bullish_div
    return out
