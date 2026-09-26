"""Orquestacion de las 4 capas de la reformulacion
(logs/reformulacion_diseno_capas.md) sobre un simbolo: D1 regimen -> razon
dominante (D1+H1, percentil expandible) -> confirmacion M15 (percentil
movil) -> geometria de entrada/gestion (capa 4). Reemplaza el flujo AND-gate
de `MultiTimeframePipeline.run_from_analyses` -- evalua AMBAS direcciones en
cada ancla (no solo la alineada con el bias D1), la leccion explicita de
Fase 7 sobre gates ocultando oportunidades reales.

No reintroduce un filtro de sesion/killzone: el diseno aprobado
(reformulacion_diseno_capas.md) no lo menciona en ninguna de las 4 capas ni
en los controles de portafolio -- agregar uno aca seria una regla nueva no
aprobada. El filtro de noticias de alto impacto SI se mantiene (no es parte
de las 4 capas de decision, es una restriccion operativa independiente ya
validada en `prompt-implementar-m15-n2.md`).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trader.confirmation import compute_confirmations
from trader.config import TraderConfig
from trader.dominant_reason import (
    DominantReasonEngine,
    _displacement_ratio_at,
    _matching_level_touches,
    _wick_ratio_at,
)
from trader.events import TF_DURATION, MarketEvent
from trader.management import BASE_RISK_PCT, conviction_multiplier, liquidity_targets, structural_buffer
from trader.news_filter import is_high_impact_news_window
from trader.pipeline.engine import TimeframeAnalysis
from trader.regime import regime_as_of
from trader.signal import ReformedSignal

# Ventana de "reciente" para un barrido de liquidez como candidato de razon
# dominante -- SIN esto, un barrido de hace años seguiria siendo "el mas
# reciente" indefinidamente si nada mas lo reemplaza, lo cual no es fiel a
# "de las razones posibles AHORA". Se reusa la misma ventana de 20
# velas/dias que la capa 1 (regimen) usa para "que ha estado haciendo este
# par ULTIMAMENTE" -- mismo horizonte temporal, no un numero nuevo sin
# relacion con el resto del diseno. Reciente por si solo NO alcanza -- ver
# `_recent_sweep`, que ademas exige que el precio actual siga cerca del
# nivel barrido (misma banda que `_overlapping_zone`/`_nearby_level`).
SWEEP_RECENCY_WINDOW = pd.Timedelta(days=20)

_LEGACY_PATTERN_KINDS = {"choch", "turtle_soup_bullish", "turtle_soup_bearish", "sharp_turn"}


@dataclass(frozen=True)
class GeneratedSignal:
    signal: ReformedSignal
    anchor_bar_idx: int  # indice en m15_full.df de la vela de entrada de referencia -- para simulate_managed_trade


def _ts_to_idx(df: pd.DataFrame) -> dict:
    return {ts: i for i, ts in enumerate(df["timestamp"])}


def build_dominant_reason_engine(
    d1_analysis: TimeframeAnalysis, h1_analysis: TimeframeAnalysis, min_sweep_wick_price: float = 0.0
) -> DominantReasonEngine:
    """Alimenta los 3 trackers UNA VEZ, recorriendo cada lista homogenea
    (sweeps, zonas, niveles) ordenada por timestamp -- D1 y H1 se combinan
    con un filtro lineal + sort explicito, nunca con bisect sobre la
    concatenacion (la regla explicita de logs/reformulacion_diseno_capas.md
    para no repetir el bug de Fase 7).

    `min_sweep_wick_price`: piso absoluto (prompt-3-chequeos-antes-de-cerrar.md,
    chequeo 1) sobre la mecha del barrido en precio -- `sweep_wick_min_pct`
    (config existente) ya exige que la mecha sea un % minimo del rango de SU
    PROPIA vela, pero eso no evita que la vela entera sea diminuta en
    terminos absolutos. Zonas (OB/FVG) ya tienen su propio piso absoluto via
    los detectores existentes (`displacement_atr_multiple`, `min_gap_pct`) --
    solo el barrido carecia de uno."""
    engine = DominantReasonEngine()
    d1_idx = _ts_to_idx(d1_analysis.df)
    h1_idx = _ts_to_idx(h1_analysis.df)

    sweeps = sorted(d1_analysis.sweeps + h1_analysis.sweeps, key=lambda e: e.timestamp)
    zones = sorted(
        d1_analysis.order_blocks + d1_analysis.fvgs + h1_analysis.order_blocks + h1_analysis.fvgs,
        key=lambda e: e.timestamp,
    )
    levels = sorted(d1_analysis.equal_levels + h1_analysis.equal_levels, key=lambda e: e.timestamp)

    for ev in sweeps:
        is_d1 = ev.timeframe == "D1"
        df = d1_analysis.df if is_d1 else h1_analysis.df
        idx_map = d1_idx if is_d1 else h1_idx
        levels_pool = d1_analysis.equal_levels if is_d1 else h1_analysis.equal_levels
        wick_ratio = _wick_ratio_at(df, idx_map, ev)
        idx = idx_map.get(ev.timestamp)
        if idx is not None:
            candle_range = float(df["high"].iloc[idx] - df["low"].iloc[idx])
            if wick_ratio * candle_range < min_sweep_wick_price:
                continue
        touches = _matching_level_touches(ev, levels_pool)
        engine.ingest_sweep(ev, wick_ratio, touches)

    for ev in zones:
        is_d1 = ev.timeframe == "D1"
        df = d1_analysis.df if is_d1 else h1_analysis.df
        idx_map = d1_idx if is_d1 else h1_idx
        atr = d1_analysis.atr if is_d1 else h1_analysis.atr
        displacement = _displacement_ratio_at(df, atr, idx_map, ev.timestamp)
        fresh = ev.confirmed_at == ev.timestamp  # "nunca tocada" -- ver Capa 2 del diseno
        engine.ingest_zone(ev, displacement, fresh)

    for ev in levels:
        touches = float(ev.meta.get("count", 1))
        # Un nivel S/R no tiene una direccion propia (es "neutral" en el
        # detector) -- puede actuar como soporte (candidato alcista) o
        # resistencia (candidato bajista) segun desde que lado se lo mire.
        # Se ingesta en AMBOS trackers con el mismo touches crudo; el
        # decaimiento por antigüedad se aplica despues, por ancla (ver
        # `_nearby_level_age_days` mas abajo), no aca.
        engine.ingest_level(ev, touches, "bullish")
        engine.ingest_level(ev, touches, "bearish")

    return engine


def _recent_sweep(
    sweeps: list[MarketEvent], direction: str, as_of: pd.Timestamp, current_price: float, tol: float
) -> MarketEvent | None:
    """Reciente (<=20 dias) Y geometricamente relevante AHORA: el precio
    actual debe seguir cerca del nivel barrido (misma banda de tolerancia
    que `_overlapping_zone`/`_nearby_level` ya usan). Sin este segundo
    filtro, cualquier barrido de los ultimos 20 dias calificaba sin importar
    cuanto se hubiera alejado el precio desde entonces -- verificado
    empiricamente contra datos reales de EURUSD: la distancia mediana entre
    el precio de entrada y el nivel barrido llegaba a 0.22% (hasta 2.86%),
    nada parecido a "la razon por la que el precio se mueve DESDE AQUI"
    (Capa 2 del diseno). Bug real encontrado al auditar por que el primer
    backtest completo devolvio n=8683 (~1000x cualquier baseline anterior de
    este proyecto) -- confirmado y corregido antes de reportar cualquier
    numero, no despues."""
    candidates = [
        e for e in sweeps
        if e.direction == direction
        and e.timestamp <= as_of
        and (as_of - e.timestamp) <= SWEEP_RECENCY_WINDOW
        and e.overlaps(current_price * (1 - tol), current_price * (1 + tol))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e.timestamp)


def _overlapping_zone(
    zones: list[MarketEvent], direction: str, as_of: pd.Timestamp, current_price: float, tol: float
) -> MarketEvent | None:
    candidates = [
        z for z in zones
        if z.direction == direction
        and z.confirmed_at is not None
        and z.confirmed_at <= as_of
        and (z.broken_at is None or as_of < z.broken_at)
        and z.overlaps(current_price * (1 - tol), current_price * (1 + tol))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e.timestamp)


def _nearby_level(levels: list[MarketEvent], as_of: pd.Timestamp, current_price: float, tol: float) -> MarketEvent | None:
    candidates = [
        lvl for lvl in levels
        if lvl.timestamp <= as_of and lvl.overlaps(current_price * (1 - tol), current_price * (1 + tol))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e.timestamp)


def _minimum_risk_floor(atr: pd.Series, atr_multiple: float) -> float | None:
    if len(atr) == 0:
        return None
    latest = atr.iloc[-1]
    if pd.isna(latest):
        return None
    return atr_multiple * float(latest)


def _classify_named_pattern(m15_trunc: TimeframeAnalysis, rejection_close_ts: pd.Timestamp, direction: str) -> str | None:
    """Solo para logging/comparacion con fases anteriores -- NUNCA parte de
    la decision (Capa 3 del diseno). Si la vela de rechazo coincide con uno
    de los 4 patrones nombrados que el catalogo cerrado anterior aceptaba,
    se etiqueta; si no, None."""
    for e in m15_trunc.all_events():
        if e.timestamp == rejection_close_ts and e.direction == direction and e.kind in _LEGACY_PATTERN_KINDS:
            return e.kind
        if e.timestamp == rejection_close_ts and e.direction == direction and e.kind.startswith("inverted_fair_value_gap"):
            return e.kind
    return None


MIN_ABSOLUTE_FLOOR_SPREAD_MULTIPLE = 3.0  # prompt-3-chequeos-antes-de-cerrar.md, chequeo 1


def generate_reformed_signals(
    symbol: str,
    d1_full: TimeframeAnalysis,
    h1_full: TimeframeAnalysis,
    m15_full: TimeframeAnalysis,
    config: TraderConfig,
    avg_spread_price: float = 0.0,
) -> list[GeneratedSignal]:
    """`avg_spread_price`: spread promedio REAL del simbolo (mismo que ya
    calcula `estimate_symbol_cost` para costos) -- deriva el piso absoluto de
    la Capa 3 y de la mecha de barrido de la Capa 2 como
    `MIN_ABSOLUTE_FLOOR_SPREAD_MULTIPLE` veces el spread propio del simbolo,
    nunca un pip/ATR fijo cross-symbol (la leccion ya aprendida con el
    umbral de FVG en la fase LTF). 0.0 (default) preserva el comportamiento
    anterior sin piso -- para tests/smoke que no tienen un spread real que
    pasar."""
    min_floor = MIN_ABSOLUTE_FLOOR_SPREAD_MULTIPLE * avg_spread_price
    confirmations = compute_confirmations(m15_full.df, min_candle_range=min_floor)
    engine = build_dominant_reason_engine(d1_full, h1_full, min_sweep_wick_price=min_floor)
    tol = config.poi.tolerance_pct / 100.0

    out: list[GeneratedSignal] = []
    n = len(m15_full.df)
    for i in range(n - 1):
        row = confirmations.get(i, {})
        if row.get("bullish") is None and row.get("bearish") is None:
            continue

        anchor_idx = i + 1
        as_of = m15_full.df["timestamp"].iloc[anchor_idx] + TF_DURATION["M15"]
        current_price = float(m15_full.df["close"].iloc[anchor_idx])
        rejection_close_ts = m15_full.df["timestamp"].iloc[i] + TF_DURATION["M15"]

        blocked, _ = is_high_impact_news_window(as_of, config.news_filter)
        if blocked:
            continue

        d1_trunc = d1_full.as_of(as_of)
        if len(d1_trunc.df) == 0:
            continue
        regime_result = regime_as_of(d1_trunc.df, len(d1_trunc.df) - 1)
        if regime_result is None:
            continue

        h1_trunc = h1_full.as_of(as_of)
        m15_trunc = m15_full.as_of(as_of)

        for event_direction in ("bullish", "bearish"):
            confirmation = row.get(event_direction)
            if confirmation is None:
                continue

            sweep_pool = [e for e in (d1_trunc.sweeps + h1_trunc.sweeps) if e.direction == event_direction]
            recent_sweep = _recent_sweep(sweep_pool, event_direction, as_of, current_price, tol)

            zone_pool = [
                z for z in (d1_trunc.order_blocks + d1_trunc.fvgs + h1_trunc.order_blocks + h1_trunc.fvgs)
                if z.direction == event_direction
            ]
            overlapping_zone = _overlapping_zone(zone_pool, event_direction, as_of, current_price, tol)

            level_pool = d1_trunc.equal_levels + h1_trunc.equal_levels
            nearby_level = _nearby_level(level_pool, as_of, current_price, tol)
            level_age_days = (as_of - nearby_level.timestamp).total_seconds() / 86400.0 if nearby_level else None

            dominant = engine.best_candidate(
                event_direction, current_price, recent_sweep, overlapping_zone, nearby_level, level_age_days,
            )
            if dominant is None:
                continue

            trade_direction = "long" if event_direction == "bullish" else "short"
            buffer = structural_buffer(m15_trunc.df)
            ref_low, ref_high = dominant.source_event.price_low, dominant.source_event.price_high
            original_sl = (ref_low - buffer) if trade_direction == "long" else (ref_high + buffer)

            risk_price = (current_price - original_sl) if trade_direction == "long" else (original_sl - current_price)
            if risk_price <= 0:
                continue
            min_risk = _minimum_risk_floor(m15_trunc.atr, config.risk.min_risk_atr_multiple)
            if min_risk is not None and risk_price < min_risk:
                continue

            targets = liquidity_targets(trade_direction, m15_trunc, current_price)
            if not targets:
                continue
            final_target = targets[-1]
            partial_target = targets[0] if len(targets) >= 2 else None

            reward_price = (final_target - current_price) if trade_direction == "long" else (current_price - final_target)
            if reward_price <= 0:
                continue
            planned_rr = reward_price / risk_price
            if planned_rr < config.risk.min_risk_reward:
                continue

            conviction = conviction_multiplier(dominant.strength_percentile, confirmation.score_percentile)
            risk_pct_applied = BASE_RISK_PCT * conviction
            named_pattern = _classify_named_pattern(m15_trunc, rejection_close_ts, event_direction)

            signal = ReformedSignal(
                symbol=symbol,
                direction=trade_direction,
                generated_at=as_of,
                regime=regime_result.regime,
                regime_efficiency_ratio=regime_result.efficiency_ratio,
                dominant_reason_kind=dominant.kind,
                dominant_reason_strength_percentile=dominant.strength_percentile,
                confirmation_score_percentile=confirmation.score_percentile,
                confirmation_followthrough=confirmation.followthrough,
                named_pattern_classification=named_pattern,
                conviction_multiplier=conviction,
                risk_pct_applied=risk_pct_applied,
                entry_reference_price=current_price,
                original_sl=original_sl,
                partial_target=partial_target,
                final_target=final_target,
                planned_risk_reward=planned_rr,
            )
            out.append(GeneratedSignal(signal=signal, anchor_bar_idx=anchor_idx))

    return out
