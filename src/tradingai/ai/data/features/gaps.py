"""Fair Value Gaps (FVG) / imbalances: desequilibrios de precio de 3 velas.

FVG alcista: low[i+1] > high[i-1]  (hueco entre la sombra de la vela 1 y la vela 3)
FVG bajista: high[i+1] < low[i-1]
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_fvg(df: pd.DataFrame, min_gap_size_pips: float = 2.0, pip_size: float = 0.0001) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    bullish_fvg = np.zeros(n, dtype=bool)
    bearish_fvg = np.zeros(n, dtype=bool)
    fvg_top = np.full(n, np.nan)
    fvg_bottom = np.full(n, np.nan)

    min_gap = min_gap_size_pips * pip_size
    high, low = out["high"].to_numpy(), out["low"].to_numpy()

    for i in range(1, n - 1):
        gap_up = low[i + 1] - high[i - 1]
        if gap_up > min_gap:
            bullish_fvg[i] = True
            fvg_bottom[i] = high[i - 1]
            fvg_top[i] = low[i + 1]

        gap_down = low[i - 1] - high[i + 1]
        if gap_down > min_gap:
            bearish_fvg[i] = True
            fvg_top[i] = low[i - 1]
            fvg_bottom[i] = high[i + 1]

    out["bullish_fvg"] = bullish_fvg
    out["bearish_fvg"] = bearish_fvg

    # Distancia al cierre en % (no precio absoluto): comparable entre simbolos y
    # regimenes de precio, y 0.0 donde no hay gap es un sentinel neutro que no
    # genera NaN (la mayoria de las velas no tienen FVG activo).
    close = out["close"].to_numpy()
    out["fvg_top_pct"] = np.where(np.isnan(fvg_top), 0.0, (fvg_top - close) / close)
    out["fvg_bottom_pct"] = np.where(np.isnan(fvg_bottom), 0.0, (fvg_bottom - close) / close)
    return out
