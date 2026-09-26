"""NUEVO -- contenedor de análisis causal propio para este paquete,
deliberadamente SEPARADO de `pipeline/engine.py::TimeframeAnalysis` para
mantener aislamiento completo del pipeline de producción (nunca se importa
nada de `trader/pipeline/` desde `mtf_strategies/`). Reutiliza exactamente
los mismos detectores de bajo nivel que `TimeframeAnalysis` (misma fuente de
verdad, `detectors/*.py`).

`build_analysis()` (recompute-desde-cero) se usó para el smoke-test rápido
del Prompt 2. Escanear una ventana histórica real (Paso 3 -- semanas/meses,
miles de anchors) con esa función resultó demasiado lento: reconstruye TODOS
los detectores desde cero en cada cutoff. `PrecomputedHistory` es la misma
idea que ya documenta `pipeline/engine.py::TimeframeAnalysis.as_of()`: cada
detector es causal por construcción (un evento en T solo depende de datos
<=T), así que correrlos UNA VEZ sobre toda la historia disponible y despues
filtrar por `timestamp<=cutoff` da el resultado IDÉNTICO a recomputar desde
cero en cada cutoff.

Order Blocks/IOB necesitan un paso más que el resto (swings/estructura/
sweeps/equal-levels/FVG, todos "calcular una vez y filtrar por timestamp"
sin nada más): `mitigated_at`/`broken_at` se determinan escaneando HACIA
ADELANTE desde la formación de cada zona. La primera versión de este archivo
volvía a correr ESE escaneo en cada `.as_of(cutoff)` sobre los OBs
pendientes -- funcionaba, pero con miles de zonas en una ventana de meses
era, medido, el costo dominante de todo el escaneo (no un problema menor).

La solución no es dejar de trackear touched/broken (la decisión de diseño
confirmada pide Order Block **e IOB** como candidatos -- un IOB solo existe
cuando una zona se rompe, así que hace falta la lógica de ruptura, no se
puede simplemente usar zonas crudas como con FVG). La solución es la misma
que ya usa `TimeframeAnalysis.as_of()::_zones()`: el PRIMER touch/break
hacia adelante desde la formación de una zona no cambia por tener más datos
futuros disponibles -- así que se calcula UNA VEZ sobre la historia
COMPLETA, y en cada cutoff simplemente se "recorta" ese touched_at/broken_at
ya conocido a `<= cutoff` (None si todavía no pasó) en vez de volver a
escanear. Mismo resultado causal exacto, sin el re-scan.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from trader.config import TraderConfig
from trader.detectors.common import INVERTED_KIND, apply_mitigation
from trader.detectors.fvg import detect_fvg
from trader.detectors.indicators import atr
from trader.detectors.liquidity import detect_equal_levels, detect_liquidity_sweeps
from trader.detectors.order_blocks import detect_order_blocks
from trader.detectors.structure import detect_structure_breaks, detect_swings
from trader.events import TF_DURATION, MarketEvent, closed_candles_as_of, ensure_utc_sorted


@dataclass(frozen=True)
class SimpleAnalysis:
    timeframe: str
    df: pd.DataFrame
    swings: list[MarketEvent]
    structure_events: list[MarketEvent]
    sweeps: list[MarketEvent]
    equal_levels: list[MarketEvent]
    fvgs: list[MarketEvent]
    order_blocks: list[MarketEvent]
    inverted_order_blocks: list[MarketEvent]
    atr: pd.Series


def build_analysis(raw_df: pd.DataFrame, timeframe: str, as_of: pd.Timestamp, config: TraderConfig) -> SimpleAnalysis:
    """Recompute-desde-cero -- correcto siempre, caro para escanear muchos
    cutoffs seguidos. Usar `PrecomputedHistory` para eso."""
    df = closed_candles_as_of(raw_df, timeframe, as_of)
    swings = detect_swings(df, timeframe, config.structure.swing_left_bars, config.structure.swing_right_bars)
    structure_events = detect_structure_breaks(df, timeframe, swings)
    sweeps = detect_liquidity_sweeps(df, timeframe, swings, config.liquidity.sweep_wick_min_pct)
    equal_levels = detect_equal_levels(swings, timeframe, config.liquidity.equal_level_tolerance_pct)
    fvgs = detect_fvg(df, timeframe, config.fvg.min_gap_pct)
    atr_series = atr(df, period=config.structure.atr_period)

    raw_obs = detect_order_blocks(df, timeframe, structure_events, atr_series, config.structure.displacement_atr_multiple)
    grace_period = TF_DURATION["M15"] * config.zone_lifecycle.invalidation_grace_m15_candles
    obs, iobs = apply_mitigation(raw_obs, df, timeframe, grace_period)

    return SimpleAnalysis(
        timeframe=timeframe, df=df, swings=swings, structure_events=structure_events,
        sweeps=sweeps, equal_levels=equal_levels, fvgs=fvgs,
        order_blocks=obs, inverted_order_blocks=iobs, atr=atr_series,
    )


class PrecomputedHistory:
    """Corre todos los detectores UNA VEZ sobre `raw_df` completo, expone
    `.as_of(cutoff)` como un filtro barato (sin recompute) para escanear
    miles de cutoffs."""

    def __init__(self, raw_df: pd.DataFrame, timeframe: str, config: TraderConfig):
        ensure_utc_sorted(raw_df)
        self.timeframe = timeframe
        self.config = config
        self.df = raw_df.reset_index(drop=True)
        self.close_ts = self.df["timestamp"] + TF_DURATION[timeframe]

        self.swings = detect_swings(self.df, timeframe, config.structure.swing_left_bars, config.structure.swing_right_bars)
        self.structure_events = detect_structure_breaks(self.df, timeframe, self.swings)
        self.sweeps = detect_liquidity_sweeps(self.df, timeframe, self.swings, config.liquidity.sweep_wick_min_pct)
        self.equal_levels = detect_equal_levels(self.swings, timeframe, config.liquidity.equal_level_tolerance_pct)
        self.fvgs = detect_fvg(self.df, timeframe, config.fvg.min_gap_pct)
        self.atr = atr(self.df, period=config.structure.atr_period)
        self.grace_period = TF_DURATION["M15"] * config.zone_lifecycle.invalidation_grace_m15_candles

        raw_obs = detect_order_blocks(self.df, timeframe, self.structure_events, self.atr, config.structure.displacement_atr_multiple)
        # touched_at/broken_at completos (historia entera) -- el PRIMER touch/
        # break hacia adelante desde la formación de una zona no cambia con
        # más datos futuros, así que esto se computa una sola vez y cada
        # `.as_of(cutoff)` solo recorta estos valores ya conocidos.
        self.full_history_obs, _ = apply_mitigation(raw_obs, self.df, timeframe, self.grace_period)

    def as_of(self, cutoff: pd.Timestamp) -> SimpleAnalysis:
        cutoff_idx = int((self.close_ts <= cutoff).sum())
        df = self.df.iloc[:cutoff_idx].reset_index(drop=True)

        def _events(events: list[MarketEvent]) -> list[MarketEvent]:
            return [e for e in events if e.timestamp <= cutoff]

        obs: list[MarketEvent] = []
        inversions: list[MarketEvent] = []
        for z in self.full_history_obs:
            if z.timestamp > cutoff:
                continue  # la zona ni siquiera se formó todavía a este cutoff
            touched_at = z.mitigated_at if (z.mitigated_at is not None and z.mitigated_at <= cutoff) else None
            broken_at = z.broken_at if (z.broken_at is not None and z.broken_at <= cutoff) else None
            if touched_at is None:
                confirmed_at = z.timestamp
            elif broken_at is None or (broken_at - touched_at) >= self.grace_period:
                confirmed_at = touched_at + self.grace_period
            else:
                confirmed_at = None
            obs.append(replace(z, mitigated=touched_at is not None, mitigated_at=touched_at, broken_at=broken_at, confirmed_at=confirmed_at))
            if broken_at is not None:
                inv_kind = INVERTED_KIND.get(z.kind)
                if inv_kind:
                    inversions.append(MarketEvent.zone(
                        kind=inv_kind, timeframe=self.timeframe, timestamp=broken_at,
                        direction="bearish" if z.direction == "bullish" else "bullish",
                        price_high=z.price_high, price_low=z.price_low, origin_timestamp=z.timestamp,
                    ))

        return SimpleAnalysis(
            timeframe=self.timeframe, df=df,
            swings=_events(self.swings), structure_events=_events(self.structure_events),
            sweeps=_events(self.sweeps), equal_levels=_events(self.equal_levels), fvgs=_events(self.fvgs),
            order_blocks=obs, inverted_order_blocks=inversions,
            atr=self.atr.iloc[:len(df)],
        )
