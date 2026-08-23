"""Smart Money Concepts: order blocks, liquidity pools, premium/discount.

Order block (simplificado): ultima vela bajista antes de un impulso alcista
fuerte (o viceversa), zona desde la que se espera reaccion del precio.

Liquidity pool: zona de equal highs / equal lows donde se acumulan stops,
objetivo tipico de "barridos de liquidez" antes de un movimiento real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_order_blocks(
    df: pd.DataFrame,
    lookback: int = 50,
    impulse_atr_mult: float = 1.5,
) -> pd.DataFrame:
    """Marca velas candidatas a order block bullish/bearish segun el impulso que las sigue."""
    out = df.copy()
    atr = _atr(out, period=14)

    bullish_ob = np.zeros(len(out), dtype=bool)
    bearish_ob = np.zeros(len(out), dtype=bool)

    body = out["close"] - out["open"]
    for i in range(len(out) - 1):
        impulse = out["close"].iloc[i + 1] - out["open"].iloc[i + 1]
        threshold = atr.iloc[i + 1] * impulse_atr_mult if not np.isnan(atr.iloc[i + 1]) else np.inf

        # Ultima vela bajista antes de impulso alcista fuerte -> order block alcista
        if body.iloc[i] < 0 and impulse > threshold:
            bullish_ob[i] = True
        # Ultima vela alcista antes de impulso bajista fuerte -> order block bajista
        if body.iloc[i] > 0 and impulse < -threshold:
            bearish_ob[i] = True

    out["bullish_ob"] = bullish_ob
    out["bearish_ob"] = bearish_ob
    return out


def detect_liquidity_pools(df: pd.DataFrame, lookback: int = 100, tolerance_pct: float = 0.05) -> pd.DataFrame:
    """Detecta equal highs / equal lows (pools de liquidez) dentro de una ventana movil."""
    out = df.copy()
    equal_high = np.zeros(len(out), dtype=bool)
    equal_low = np.zeros(len(out), dtype=bool)

    for i in range(lookback, len(out)):
        window_high = out["high"].iloc[i - lookback : i]
        window_low = out["low"].iloc[i - lookback : i]
        h, l = out["high"].iloc[i], out["low"].iloc[i]

        if (np.abs(window_high - h) / h * 100 < tolerance_pct).sum() >= 2:
            equal_high[i] = True
        if (np.abs(window_low - l) / l * 100 < tolerance_pct).sum() >= 2:
            equal_low[i] = True

    out["equal_high"] = equal_high
    out["equal_low"] = equal_low
    return out


def premium_discount_zone(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """Ubica el precio dentro del rango [0,1] del swing reciente: <0.5 discount, >0.5 premium."""
    out = df.copy()
    range_high = out["high"].rolling(lookback).max()
    range_low = out["low"].rolling(lookback).min()
    out["pd_zone"] = (out["close"] - range_low) / (range_high - range_low)
    return out


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()
