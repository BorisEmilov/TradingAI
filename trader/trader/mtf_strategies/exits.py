"""NUEVO -- un único criterio temporal, reutilizado en DOS lugares distintos
por decisión de diseño explícita (no dos parámetros independientes):

1. Salida temporal post-entrada (regla 14 / Estrategia 2 regla 9): si pasan
   `MAX_CANDLES_M15` velas de 15M desde la ENTRADA sin alcanzar +0.5R, cerrar.
2. Expiración del setup si el precio nunca vuelve al 50% de la FVG (regla 8,
   sin número propio en la spec original) -- mismo umbral, contado desde que
   el MSS/FVG quedó confirmado, no desde la entrada (todavía no hay entrada
   en ese punto).

`MAX_CANDLES_M15=6` (~90 min) es un parámetro A VALIDAR EN BACKTEST, no
asumido óptimo -- lo dice la spec explícitamente, se deja como constante
fácil de barrer, no hardcodeado disperso por el código.
"""

from __future__ import annotations

import pandas as pd

MAX_CANDLES_M15 = 6
MAX_MINUTES_M15 = 90
MIN_PROGRESS_R = 0.5


def candles_elapsed_m15(reference_time: pd.Timestamp, current_time: pd.Timestamp) -> int:
    if current_time <= reference_time:
        return 0
    return int((current_time - reference_time) / pd.Timedelta(minutes=15))


def temporal_exit_triggered(reference_time: pd.Timestamp, current_time: pd.Timestamp, progress_r: float) -> bool:
    """Post-entrada: cerrar si pasaron 6 velas (90 min) sin +0.5R."""
    return candles_elapsed_m15(reference_time, current_time) >= MAX_CANDLES_M15 and progress_r < MIN_PROGRESS_R


def fvg_setup_expired(fvg_confirmed_at: pd.Timestamp, current_time: pd.Timestamp, retraced_to_50pct: bool) -> bool:
    """Pre-entrada: el setup (MSS+FVG confirmados, esperando el retroceso al
    50%) se descarta si pasaron 6 velas / 90 min sin que el precio haya
    vuelto todavía -- mismo umbral que `temporal_exit_triggered`, sin
    condición de +0.5R (todavía no hay posición abierta de la cual medir R)."""
    if retraced_to_50pct:
        return False
    return candles_elapsed_m15(fvg_confirmed_at, current_time) >= MAX_CANDLES_M15
