"""ADAPTADO de `detectors/structure.py::detect_structure_breaks` -- reutiliza
esa función sin tocarla, pero acotada explícitamente a los swings formados
DESPUÉS del sweep de 1H (decisión de diseño confirmada: "no mires la serie
completa"). `detect_structure_breaks` ya funciona así si se le pasa una lista
de swings pre-filtrada -- itera todo el DataFrame pero solo empieza a trackear
`last_swing_high`/`last_swing_low` a medida que aparecen swings en la lista
que se le pasa, así que filtrar los swings de entrada alcanza para acotar el
MSS a la micro-estructura post-sweep, sin reimplementar la lógica de ruptura.

MSS != BOS/CHoCH genérico: aquí importa específicamente la PRIMERA ruptura de
estructura que ocurre después del sweep, en la dirección de la reversión
esperada (alcista tras un sweep bajista, bajista tras un sweep alcista) --
`first_post_sweep_mss()` filtra por eso, no devuelve cualquier ruptura.
"""

from __future__ import annotations

import pandas as pd

from trader.detectors.structure import detect_structure_breaks
from trader.events import TF_DURATION, MarketEvent


def first_post_sweep_mss(
    m15_df: pd.DataFrame, m15_swings: list[MarketEvent], sweep_timestamp: pd.Timestamp, expected_direction: str
) -> MarketEvent | None:
    """`expected_direction`: "bullish" tras un sweep de liquidez inferior
    (buscamos MSS alcista = ruptura del último lower high post-sweep),
    "bearish" tras un sweep de liquidez superior."""
    post_sweep_swings = [s for s in m15_swings if s.timestamp > sweep_timestamp]
    if not post_sweep_swings:
        return None

    # recorta `m15_df` a las velas que pueden cerrar en o después del sweep --
    # `detect_structure_breaks` itera TODO el df que se le pase sin importar
    # cuántos swings tenga la lista filtrada; pasarle años de velas
    # irrelevantes (todo lo anterior al sweep) es puro desperdicio de tiempo,
    # no cambia el resultado (nada antes del sweep puede producir un MSS
    # "posterior al sweep" de todas formas).
    relevant = m15_df[m15_df["timestamp"] + TF_DURATION["M15"] >= sweep_timestamp].reset_index(drop=True)

    breaks = detect_structure_breaks(relevant, "M15", post_sweep_swings)
    candidates = [b for b in breaks if b.timestamp > sweep_timestamp and b.direction == expected_direction]
    if not candidates:
        return None
    return min(candidates, key=lambda e: e.timestamp)  # el PRIMERO tras el sweep, no el más reciente
