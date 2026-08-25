"""Metodo Wyckoff: springs y upthrusts sobre un rango de trading detectado.

Spring: el precio barre por debajo del soporte del rango reciente (atrapa vendedores/
stops) y CIERRA de vuelta dentro del rango -> confirmacion clasica de acumulacion antes
de un movimiento alcista.
Upthrust: lo simetrico sobre la resistencia del rango -> confirmacion de distribucion.

Distinto de `smc.detect_liquidity_sweeps` en la escala de referencia: liquidity sweep
mira niveles puntuales de equal-high/low (pivotes), esto mira el rango de negociacion
mas amplio (rolling high/low de una ventana mayor) -> señal complementaria, no
redundante. El rango de referencia se calcula con `.shift(1)` para excluir la vela
actual y evitar auto-referencia (la vela actual nunca "rompe" su propio extremo).
"""

from __future__ import annotations

import pandas as pd


def detect_wyckoff_events(df: pd.DataFrame, range_window: int = 20, tolerance_pct: float = 0.05) -> pd.DataFrame:
    out = df.copy()

    range_high = out["high"].rolling(range_window).max().shift(1)
    range_low = out["low"].rolling(range_window).min().shift(1)

    spring = (out["low"] < range_low * (1 - tolerance_pct / 100)) & (out["close"] > range_low)
    upthrust = (out["high"] > range_high * (1 + tolerance_pct / 100)) & (out["close"] < range_high)

    out["wyckoff_spring"] = spring.fillna(False)
    out["wyckoff_upthrust"] = upthrust.fillna(False)
    return out
