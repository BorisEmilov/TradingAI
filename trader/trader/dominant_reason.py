"""Capa 2 de la reformulacion: razon dominante. 3 candidatos (liquidez
tomada, zona HTF, nivel estructural clave), cada uno puntuado por percentil
EXPANDIBLE causal contra su propia historia (`trader/percentile.py`). Se
toma el MAXIMO entre los 3, no la suma -- percentil >=80 para contar como
"suficientemente fuerte por si sola" (ver logs/reformulacion_diseno_capas.md).

Simplificaciones documentadas explicitamente (no ocultas):
- "Touches del nivel barrido" para liquidez tomada: se busca el
  `equal_levels` mas cercano en PRECIO (no por timestamp exacto, ese dato no
  se guarda en el evento de sweep) dentro de una tolerancia -- si no hay
  ninguno cerca, se asume 1 toque (el swing barrido en si mismo).
- Decaimiento por antigüedad de nivel estructural: el nivel se ingesta UNA
  SOLA VEZ (en su formacion) con sus `touches` crudos -- igual que liquidez y
  zonas HTF, para no distorsionar el percentil expandible con el mismo nivel
  insertado muchas veces mientras el precio ronda cerca de el. El
  decaimiento `pct / (1 + edad_dias/180)` (a ~180 dias, la influencia se
  reduce a la mitad; no calibrado contra resultados, la forma mas simple de
  "menos relevante cuanto mas viejo" sin inventar una curva sin evidencia que
  la respalde) se aplica DESPUES, sobre el percentil ya cacheado, en el
  momento en que se evalua como candidato en una ancla concreta -- ver
  `trader/pipeline/reformed.py`. Aplicarlo al valor crudo ANTES del
  percentil habria exigido reinsertar el mismo nivel en cada ancla donde
  aparece como cercano, inflando su representacion en la muestra historica.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trader.events import MarketEvent
from trader.percentile import ExpandingPercentileTracker

DOMINANT_REASON_THRESHOLD = 80.0
KEY_LEVEL_DECAY_HALFLIFE_DAYS = 180.0


def _event_key(event: MarketEvent) -> tuple:
    """Stable, content-based identity for caching -- NOT `id(event)`.
    `TimeframeAnalysis.as_of()` returns `dataclasses.replace()`-built copies
    of order_blocks/fvgs whenever their causally-truncated
    confirmed_at/broken_at/mitigated_at differ from the full-history values
    (see engine.py's `_zones()`) -- a different object, same zone. Caching by
    `id()` would silently miss the percentile already computed for that zone
    the moment truncation produces a copy, which is exactly the kind of
    look-ahead-adjacent bug this project has hit before with naive identity
    assumptions across `as_of()` boundaries. `kind`/`timeframe`/`timestamp`/
    `price_high`/`price_low`/`direction` are never touched by `replace()`, so
    they're a safe, stable key.
    """
    return (event.kind, event.timeframe, event.timestamp, event.price_high, event.price_low, event.direction)


_ZONE_TYPE_RANK = {
    "order_block_bullish": 2, "order_block_bearish": 2,
    "fair_value_gap_bullish": 1, "fair_value_gap_bearish": 1,
}
_LEVEL_PRICE_TOLERANCE_PCT = 0.05  # reusa la misma tolerancia de equal_levels ya validada


@dataclass(frozen=True)
class DominantReasonCandidate:
    kind: str  # "liquidity_taken" | "htf_zone" | "key_level"
    strength_percentile: float
    source_event: MarketEvent


def _wick_ratio_at(df: pd.DataFrame, close_ts_to_idx: dict, event: MarketEvent) -> float:
    idx = close_ts_to_idx.get(event.timestamp)
    if idx is None:
        return 0.0
    o, h, l, c = df["open"].iloc[idx], df["high"].iloc[idx], df["low"].iloc[idx], df["close"].iloc[idx]
    candle_range = h - l
    if candle_range <= 0:
        return 0.0
    if event.direction == "bearish":  # sweep hacia arriba, rechazo hacia abajo
        wick = h - max(o, c)
    else:
        wick = min(o, c) - l
    return max(0.0, float(wick) / float(candle_range))


def _matching_level_touches(sweep: MarketEvent, equal_levels: list[MarketEvent]) -> int:
    best = None
    best_dist = None
    for lvl in equal_levels:
        if lvl.timestamp > sweep.timestamp:
            continue
        dist = abs(lvl.price - sweep.price) / sweep.price * 100.0 if sweep.price else float("inf")
        if dist <= _LEVEL_PRICE_TOLERANCE_PCT and (best_dist is None or dist < best_dist):
            best, best_dist = lvl, dist
    if best is None:
        return 1
    return int(best.meta.get("count", 1))


def _displacement_ratio_at(df: pd.DataFrame, atr: pd.Series, close_ts_to_idx: dict, timestamp: pd.Timestamp) -> float:
    idx = close_ts_to_idx.get(timestamp)
    if idx is None or idx >= len(atr):
        return 0.0
    atr_val = atr.iloc[idx]
    if pd.isna(atr_val) or atr_val <= 0:
        return 0.0
    candle_range = float(df["high"].iloc[idx] - df["low"].iloc[idx])
    return candle_range / float(atr_val)


class DominantReasonEngine:
    """Mantiene los 3 trackers de percentil expandible por simbolo+direccion,
    y produce el candidato con mayor percentil (o None si ninguno llega al
    umbral) para cada punto de decision. Se alimenta incrementalmente,
    recorriendo eventos EN ORDEN TEMPORAL -- ver trader/percentile.py para
    el contrato de causalidad.
    """

    def __init__(self):
        self._liquidity_tracker: dict[str, ExpandingPercentileTracker] = {"bullish": ExpandingPercentileTracker(), "bearish": ExpandingPercentileTracker()}
        self._zone_tracker: dict[str, ExpandingPercentileTracker] = {"bullish": ExpandingPercentileTracker(), "bearish": ExpandingPercentileTracker()}
        self._level_tracker: dict[str, ExpandingPercentileTracker] = {"bullish": ExpandingPercentileTracker(), "bearish": ExpandingPercentileTracker()}
        # cache de percentiles ya calculados por evento (clave estable por
        # contenido, ver _event_key -- NUNCA id(event)) para no recalcular
        self._liquidity_pct: dict[tuple, float] = {}
        self._zone_pct: dict[tuple, float] = {}
        self._level_pct: dict[tuple, float] = {}

    def ingest_sweep(self, event: MarketEvent, wick_ratio: float, touches: int) -> None:
        raw = touches * wick_ratio
        pct = self._liquidity_tracker[event.direction].percentile_then_insert(raw)
        if pct is not None:
            self._liquidity_pct[_event_key(event)] = pct

    def ingest_zone(self, event: MarketEvent, displacement_ratio: float, fresh: bool) -> None:
        # el tipo de zona (OB>FVG) y frescura desempatan DENTRO del mismo
        # percentil (redondeado), no se suman como puntos aparte -- se
        # codifican en el valor bruto ranked de forma que ambos casos con el
        # mismo desplazamiento pero distinto tipo/frescura no colisionen
        # exactamente, preservando el orden de desplazamiento como criterio
        # principal.
        type_rank = _ZONE_TYPE_RANK.get(event.kind, 0)
        fresh_bonus = 0.01 if fresh else 0.0
        raw = displacement_ratio + type_rank * 1e-6 + fresh_bonus * 1e-6
        pct = self._zone_tracker[event.direction].percentile_then_insert(raw)
        if pct is not None:
            self._zone_pct[_event_key(event)] = pct

    def ingest_level(self, event: MarketEvent, touches: float, direction: str) -> None:
        pct = self._level_tracker[direction].percentile_then_insert(touches)
        if pct is not None:
            self._level_pct[_event_key(event) + (direction,)] = pct

    def best_candidate(
        self,
        direction: str,
        current_price: float,
        recent_sweep: MarketEvent | None,
        overlapping_zone: MarketEvent | None,
        nearby_level: MarketEvent | None,
        nearby_level_age_days: float | None = None,
    ) -> DominantReasonCandidate | None:
        """`nearby_level_age_days`: dias transcurridos desde la formacion de
        `nearby_level` hasta el punto de decision actual -- aplica el
        decaimiento por antigüedad (ver docstring del modulo) al percentil ya
        cacheado de ESTE candidato especifico, antes de compararlo con los
        otros dos. None (el default) no aplica decaimiento -- solo se pasa
        cuando hay, de hecho, un `nearby_level`."""
        candidates = []
        if recent_sweep is not None:
            pct = self._liquidity_pct.get(_event_key(recent_sweep))
            if pct is not None:
                candidates.append(DominantReasonCandidate("liquidity_taken", pct, recent_sweep))
        if overlapping_zone is not None:
            pct = self._zone_pct.get(_event_key(overlapping_zone))
            if pct is not None:
                candidates.append(DominantReasonCandidate("htf_zone", pct, overlapping_zone))
        if nearby_level is not None:
            pct = self._level_pct.get(_event_key(nearby_level) + (direction,))
            if pct is not None:
                if nearby_level_age_days is not None:
                    pct = pct / (1.0 + nearby_level_age_days / KEY_LEVEL_DECAY_HALFLIFE_DAYS)
                candidates.append(DominantReasonCandidate("key_level", pct, nearby_level))
        if not candidates:
            return None
        best = max(candidates, key=lambda c: c.strength_percentile)
        if best.strength_percentile < DOMINANT_REASON_THRESHOLD:
            return None
        return best
