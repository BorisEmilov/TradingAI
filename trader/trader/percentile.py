"""Utilidades de percentil causal compartidas por las capas 2 y 3 de la
reformulacion (`logs/reformulacion_diseno_capas.md`, seccion "Causalidad de
los percentiles"). Dos variantes:

- `ExpandingPercentileTracker`: ventana desde el inicio de los datos hasta
  `t` -- para razon dominante (liquidez tomada, zona HTF, nivel clave),
  eventos raros donde se necesita toda la muestra causal disponible.
- `RollingPercentileTracker`: ventana movil de tamano fijo -- para
  confirmacion de price action (mecha/cierre), eventos frecuentes donde
  conviene mantenerse adaptado a la volatilidad actual, no diluido por anios
  de historia.

Contrato de causalidad en ambos: el percentil de un valor se calcula ANTES
de insertarlo en la muestra -- un evento nunca se compara contra si mismo,
y nada con timestamp posterior puede entrar en el calculo (el llamador
recorre los eventos en orden temporal estricto y llama
`percentile_then_insert` exactamente una vez por evento, en orden).
"""

from __future__ import annotations

import bisect
from collections import deque

MIN_EXPANDING_SAMPLE = 30


def _percentile_of_score(sorted_history: list[float], value: float) -> float:
    idx = bisect.bisect_left(sorted_history, value)
    return idx / len(sorted_history) * 100.0


class ExpandingPercentileTracker:
    def __init__(self, min_sample: int = MIN_EXPANDING_SAMPLE):
        self._sorted: list[float] = []
        self._min_sample = min_sample

    def percentile_then_insert(self, value: float) -> float | None:
        pct = _percentile_of_score(self._sorted, value) if len(self._sorted) >= self._min_sample else None
        bisect.insort(self._sorted, value)
        return pct

    def __len__(self) -> int:
        return len(self._sorted)


class RollingPercentileTracker:
    def __init__(self, window: int):
        self.window = window
        self._queue: deque[float] = deque()
        self._sorted: list[float] = []

    def percentile_then_insert(self, value: float) -> float | None:
        pct = _percentile_of_score(self._sorted, value) if len(self._sorted) >= self.window else None
        bisect.insort(self._sorted, value)
        self._queue.append(value)
        if len(self._queue) > self.window:
            oldest = self._queue.popleft()
            self._sorted.remove(oldest)
        return pct
