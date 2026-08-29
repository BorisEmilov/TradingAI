"""Fibonacci OTE (Optimal Trade Entry): zona de retroceso 61.8%-78.6% del ultimo
impulso de precio (swing a swing), concepto central de ICT/SMC.

Requiere que `market_structure.detect_swings()` ya haya corrido antes (columnas
`swing_high`/`swing_low`) -- mismo requisito que ya tiene `divergence.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_OTE_LOW = 0.618
_OTE_HIGH = 0.786


def compute_ote_zone(df: pd.DataFrame, lookback: int = 100) -> pd.DataFrame:
    """Marca si el precio esta en la zona dorada de retroceso del ultimo impulso.

    Impulso alcista (el ultimo swing_low confirmado es mas antiguo que el ultimo
    swing_high confirmado): la zona OTE para buscar continuacion alcista es el
    retroceso hacia abajo, entre 61.8% y 78.6% de ese tramo (medido desde el
    high). Simetrico para impulso bajista (retroceso hacia arriba, medido desde
    el low). `fib_retracement_pct` es la version continua (0 = sin retroceso en
    el extremo del impulso, 1 = retroceso completo al origen), disponible aunque
    no caiga en la banda dorada, para que el modelo pueda aprender otros umbrales.
    """
    out = df.copy()
    if "swing_high" not in out.columns or "swing_low" not in out.columns:
        raise ValueError("Ejecuta market_structure.detect_swings() antes de compute_ote_zone().")

    n = len(out)
    high, low, close = out["high"].to_numpy(), out["low"].to_numpy(), out["close"].to_numpy()
    swing_high, swing_low = out["swing_high"].to_numpy(), out["swing_low"].to_numpy()

    in_ote_bullish = np.zeros(n, dtype=bool)
    in_ote_bearish = np.zeros(n, dtype=bool)
    fib_retracement_pct = np.zeros(n, dtype=float)

    last_swing_high_idx = -1
    last_swing_low_idx = -1

    for i in range(n):
        if swing_high[i]:
            last_swing_high_idx = i
        if swing_low[i]:
            last_swing_low_idx = i

        if last_swing_high_idx == -1 or last_swing_low_idx == -1:
            continue
        if i - max(last_swing_high_idx, last_swing_low_idx) > lookback:
            continue  # el impulso mas reciente ya quedo fuera de la ventana

        leg_high = high[last_swing_high_idx]
        leg_low = low[last_swing_low_idx]
        if leg_high <= leg_low:
            continue
        leg_range = leg_high - leg_low

        if last_swing_low_idx < last_swing_high_idx:
            retracement = (leg_high - close[i]) / leg_range
            fib_retracement_pct[i] = retracement
            if _OTE_LOW <= retracement <= _OTE_HIGH:
                in_ote_bullish[i] = True
        elif last_swing_high_idx < last_swing_low_idx:
            retracement = (close[i] - leg_low) / leg_range
            fib_retracement_pct[i] = retracement
            if _OTE_LOW <= retracement <= _OTE_HIGH:
                in_ote_bearish[i] = True

    out["in_ote_bullish"] = in_ote_bullish
    out["in_ote_bearish"] = in_ote_bearish
    out["fib_retracement_pct"] = fib_retracement_pct
    return out
