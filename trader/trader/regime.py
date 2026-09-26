"""Capa 1 de la reformulacion (logs/reformulacion_diseno_capas.md): clasifica
cada simbolo como tendencia o rango ANTES de buscar cualquier setup.

Metrica: razon de eficiencia sobre D1, ventana de 20 velas (~1 mes de
trading) -- `|movimiento neto| / distancia total recorrida`. Sin ATR ni
ningun indicador nuevo: aritmetica directa sobre cierres, auto-normalizada.
Umbral 0.5 (punto medio simetrico, no calibrado contra resultados) separa
tendencia de rango.

Causal por construccion: `regime_as_of(df, idx)` solo mira `df` hasta `idx`
inclusive -- nunca hace falta truncar nada aparte, el llamador ya decide
hasta que vela D1 se puede ver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WINDOW = 20
EFFICIENCY_THRESHOLD = 0.5


@dataclass(frozen=True)
class RegimeResult:
    regime: str  # "trend_up" | "trend_down" | "range"
    efficiency_ratio: float


def regime_as_of(d1_df: pd.DataFrame, idx: int) -> RegimeResult | None:
    """`idx` es la posicion (0-indexed) de la ultima vela D1 YA CERRADA
    visible en este punto -- el llamador es responsable de pasar el indice
    correcto (causal), esta funcion no filtra por timestamp por si misma.
    Devuelve None si no hay suficiente historia (menos de WINDOW velas antes
    de idx) -- "sin regimen todavia", no "rango por defecto".
    """
    if idx < WINDOW:
        return None
    closes = d1_df["close"].to_numpy()[idx - WINDOW : idx + 1]
    net = closes[-1] - closes[0]
    path = np.abs(np.diff(closes)).sum()
    if path <= 0:
        return RegimeResult(regime="range", efficiency_ratio=0.0)
    efficiency = abs(net) / path
    if efficiency > EFFICIENCY_THRESHOLD:
        return RegimeResult(regime="trend_up" if net > 0 else "trend_down", efficiency_ratio=float(efficiency))
    return RegimeResult(regime="range", efficiency_ratio=float(efficiency))
