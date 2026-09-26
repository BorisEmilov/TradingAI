"""NUEVO -- `risk/levels.py::compute_trade_levels` ya hace el gate de R:R
mínimo, reutilizado tal cual abajo, pero asume un target ÚNICO ya elegido por
el llamador. Lo nuevo acá es CÓMO se elige ese target: una lista de niveles
en orden de preferencia explícito (regla 10), no "el nivel más cercano" como
hace `pipeline/engine.py::_next_liquidity_target`.

Selección: el PRIMER nivel de la lista de prioridad que queda del lado
correcto del precio (más allá del entry, en la dirección del trade) es TP2 --
no se prueba el segundo de la lista si el primero no alcanza R:R>=2; eso lo
decide la regla 11 aparte (`compute_trade_levels` devuelve None y el setup se
cancela, "NO TRADE", no se reintenta con el siguiente nivel de la lista).
"""

from __future__ import annotations

from trader.risk.levels import TradeLevels, compute_trade_levels


def select_tp2(direction: str, priority_levels: list[float | None], entry: float) -> float | None:
    """`priority_levels` ya viene en el orden de preferencia de la regla 10
    (p.ej. [PDH, PWH, SwingHigh4H, EqualHigh, SwingHigh1H] para LONG) --
    entradas `None` (nivel no disponible, ej. PWH sin 2 semanas de historia)
    se saltan sin romper el orden de las siguientes."""
    for level in priority_levels:
        if level is None:
            continue
        if direction == "long" and level > entry:
            return level
        if direction == "short" and level < entry:
            return level
    return None


def compute_levels_with_priority_tp2(
    direction: str, entry: float, structural_stop: float, priority_levels: list[float | None], min_rr: float
) -> TradeLevels | None:
    tp2 = select_tp2(direction, priority_levels, entry)
    if tp2 is None:
        return None
    return compute_trade_levels(direction, entry, structural_stop, tp2, min_rr)
