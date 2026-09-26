"""NUEVO por completo -- Estrategia 2, regla 8: filtro de extensión, hipótesis
a testear (no asumida), umbral por defecto 1x pero pensado para barrerse.

`SwingOrigin` (decisión de diseño confirmada): el swing de signo OPUESTO que
inició la pierna actual hacia el extremo -- para un extremo superior (precio
en un swing high reciente), el origin es el ÚLTIMO swing_low anterior a ese
high; para un extremo inferior, el último swing_high anterior. Reutiliza
`mtf_strategies.bias._alternating` para no reimplementar el colapso de
corridas del mismo tipo.
"""

from __future__ import annotations

from trader.events import MarketEvent
from trader.mtf_strategies.bias import _alternating

DEFAULT_EXTENSION_THRESHOLD = 1.0


def find_swing_origin(swings: list[MarketEvent], extremity: MarketEvent) -> MarketEvent | None:
    """`extremity` es el swing (high o low) que marca el extremo 4H actual.
    El origin es el swing de tipo opuesto INMEDIATAMENTE anterior a él en la
    secuencia alternada -- None si no hay ninguno (historia insuficiente)."""
    alt = _alternating(swings)
    opposite_kind = "swing_low" if extremity.kind == "swing_high" else "swing_high"
    prior = [s for s in alt if s.timestamp < extremity.timestamp and s.kind == opposite_kind]
    if not prior:
        return None
    return max(prior, key=lambda e: e.timestamp)


def compute_extension(current_price: float, swing_origin_price: float, atr_4h: float) -> float | None:
    if atr_4h is None or atr_4h <= 0:
        return None
    return abs(current_price - swing_origin_price) / atr_4h


def passes_extension_filter(current_price: float, swing_origin_price: float, atr_4h: float, threshold: float = DEFAULT_EXTENSION_THRESHOLD) -> bool:
    ext = compute_extension(current_price, swing_origin_price, atr_4h)
    return ext is not None and ext > threshold
