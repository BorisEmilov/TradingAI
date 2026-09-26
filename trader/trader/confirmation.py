"""Capa 3 de la reformulacion: confirmacion de price action amplia. Sin
catalogo cerrado de patrones nombrados -- ratio de mecha de rechazo +
posicion del cierre, percentil MOVIL causal (ultimas 500 velas cerradas
antes de `t`, ver trader/percentile.py) contra la propia historia reciente
del simbolo/timeframe de ejecucion. Umbral >=70, MAS momentum de
confirmacion obligatorio (la vela siguiente debe cerrar mas alla del cierre
de la vela de rechazo, en la direccion favorable).

Piso absoluto (`min_candle_range`, prompt-3-chequeos-antes-de-cerrar.md,
chequeo 1): las proporciones de mecha/cierre son puramente relativas al
rango de LA MISMA vela -- una vela de 0.4 pips con mecha limpia puntua
identico a una de 8 pips con la misma forma, aunque la primera sea
indistinguible del spread. Verificado empiricamente contra EURUSD real: sin
piso, ~5.4% de las velas de confirmacion calificadas tenian un rango menor a
2 pips (comparable al spread tipico de 0.3 pips). El piso se expresa en
precio absoluto (el llamador lo deriva de `avg_spread_price` del simbolo,
nunca un pip/ATR fijo cross-symbol -- misma leccion ya aprendida con el
umbral minimo de FVG en la fase LTF) y EXCLUYE la vela por completo (no
entra al tracker ni puede confirmar) en vez de solo penalizarla -- misma
disciplina que el umbral minimo de FVG.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trader.percentile import RollingPercentileTracker

CONFIRMATION_THRESHOLD = 70.0
ROLLING_WINDOW = 500

_CONFIRMATION_KINDS_LEGACY = {"choch", "turtle_soup_bullish", "turtle_soup_bearish", "sharp_turn"}


@dataclass(frozen=True)
class ConfirmationResult:
    direction: str
    score_percentile: float
    followthrough: bool
    named_pattern: str | None  # clasificacion post-hoc, solo logging


def _wick_and_close_position(o: float, h: float, l: float, c: float, direction: str) -> tuple[float, float]:
    candle_range = h - l
    if candle_range <= 0:
        return 0.0, 0.5
    if direction == "bullish":
        wick_ratio = (min(o, c) - l) / candle_range
        close_position = (c - l) / candle_range
    else:
        wick_ratio = (h - max(o, c)) / candle_range
        close_position = (h - c) / candle_range
    return max(0.0, wick_ratio), max(0.0, min(1.0, close_position))


def compute_confirmations(
    df: pd.DataFrame, min_candle_range: float = 0.0
) -> dict[int, dict[str, ConfirmationResult]]:
    """Recorre TODAS las velas de `df` en orden, calculando para cada una
    (en ambas direcciones) su score de rechazo via percentil movil causal, y
    si la vela SIGUIENTE confirma con momentum. Devuelve
    {indice_de_la_vela_de_rechazo: {"bullish": ConfirmationResult|None, "bearish": ...}}.
    El "indice de senal" real (cuando se sabria que hubo confirmacion) es
    `indice_de_la_vela_de_rechazo + 1` -- el llamador es responsable de usar
    el timestamp de esa vela siguiente como `as_of` de la senal, nunca el de
    la vela de rechazo misma (todavia no se sabe si hubo followthrough en
    ese momento).

    `min_candle_range`: piso absoluto en precio (ver docstring del modulo).
    Una vela por debajo del piso NUNCA confirma Y NUNCA entra al tracker de
    percentil de ninguna direccion -- ni cuenta como candidata, ni contamina
    la muestra historica contra la que se comparan las demas.
    """
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    n = len(df)

    trackers = {"bullish": RollingPercentileTracker(ROLLING_WINDOW), "bearish": RollingPercentileTracker(ROLLING_WINDOW)}
    results: dict[int, dict[str, ConfirmationResult | None]] = {}

    for i in range(n):
        row_result: dict[str, ConfirmationResult | None] = {}
        if (h[i] - l[i]) < min_candle_range:
            results[i] = {"bullish": None, "bearish": None}
            continue
        for direction in ("bullish", "bearish"):
            wick_ratio, close_position = _wick_and_close_position(o[i], h[i], l[i], c[i], direction)
            raw = (wick_ratio + close_position) / 2.0
            pct = trackers[direction].percentile_then_insert(raw)

            followthrough = False
            if i + 1 < n:
                if direction == "bullish":
                    followthrough = c[i + 1] > c[i]
                else:
                    followthrough = c[i + 1] < c[i]

            if pct is not None and pct >= CONFIRMATION_THRESHOLD and followthrough:
                row_result[direction] = ConfirmationResult(
                    direction=direction, score_percentile=pct, followthrough=True,
                    named_pattern=None,  # clasificado post-hoc por separado, ver classify_named_pattern
                )
            else:
                row_result[direction] = None
        results[i] = row_result

    return results
