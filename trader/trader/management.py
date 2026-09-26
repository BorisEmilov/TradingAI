"""Capa 4 de la reformulacion: gestion basada en estructura + tamano por
conviccion (logs/reformulacion_diseno_capas.md).

Reemplaza el TP fijo + parcial en % del sistema anterior por:
- SL que SOLO se mueve ante un BOS/CHoCH real a favor de la operacion (nunca
  por tiempo ni por avance de precio en si) -- el `broken_level` que ya trae
  el evento de estructura (trader/detectors/structure.py) ES el swing
  precedente, no hace falta volver a buscarlo.
- TP final: el nivel de liquidez MAS LEJANO disponible entre los candidatos
  (soporte/resistencia, equal levels, swings) -- mismo rol que el TP del
  sistema anterior (gate de R:R en la elegibilidad de la senal Y objetivo de
  salida real durante la gestion).
- Toma parcial en el PRIMER nivel de liquidez intermedio (el mas cercano)
  ENTRE la entrada y ese TP final -- solo existe si hay al menos un segundo
  nivel mas alla de el (si el TP final YA ES el nivel mas cercano
  disponible, no hay nada "intermedio" antes de el, y no hay parcial: la
  posicion corre entera con el SL de estructura como unica proteccion hasta
  el TP o la invalidacion).
- Multiplicador de conviccion: SOLO afecta el tamano de posicion, nunca el
  resultado en R de una operacion individual (R se define contra el riesgo
  ORIGINAL, ver `_price_to_r`) -- por eso las metricas de expectancy_r/profit
  factor de este proyecto son invariantes al sizing por diseno. Se calcula y
  se registra para trazabilidad, no para alterar el backtest.

Opera sobre M15 (el unico timeframe de ejecucion de la arquitectura
reformulada -- ver dominant_reason.py/confirmation.py), a diferencia del
`simulate_trade` anterior que gestionaba contra eventos de H1 porque H1 era
la POI de la jerarquia de 2 etapas ya eliminada.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trader.backtest.costs import SymbolCost, cost_in_r
from trader.events import TF_DURATION
from trader.pipeline.engine import TimeframeAnalysis

MAX_HOLDING = pd.Timedelta(days=3)
STRUCTURAL_BUFFER_LOOKBACK = 20
STRUCTURAL_BUFFER_FACTOR = 0.1

CONVICTION_MIN = 1.0
CONVICTION_MAX = 2.0
DOMINANT_REASON_THRESHOLD = 80.0
CONFIRMATION_THRESHOLD = 70.0
BASE_RISK_PCT = 0.5  # % del capital -- logs/reformulacion_diseno_capas.md, tabla de controles de portafolio


@dataclass(frozen=True)
class ManagedTradeResult:
    symbol: str
    direction: str
    regime: str
    dominant_reason_kind: str
    entry_time: pd.Timestamp
    entry: float
    original_sl: float
    final_sl: float
    partial_target: float | None
    final_target: float
    partial_taken: bool
    sl_ratchets: int
    exit_time: pd.Timestamp
    exit_bar_index: int
    exit_reason: str
    gross_r: float
    cost_r: float
    net_r: float
    conviction_multiplier: float
    risk_pct_applied: float


def structural_buffer(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return 0.0001
    recent_range = (df["high"] - df["low"]).tail(STRUCTURAL_BUFFER_LOOKBACK).mean()
    return float(recent_range) * STRUCTURAL_BUFFER_FACTOR


def liquidity_targets(direction: str, m15_analysis: TimeframeAnalysis, entry: float) -> list[float]:
    """Niveles de liquidez por encima (long) o debajo (short) del entry,
    ordenados de mas cercano a mas lejano. El ULTIMO es el TP final; el
    PRIMERO (si hay mas de uno) es el objetivo del parcial."""
    candidates: list[float] = []
    candidates.extend(e.price for e in m15_analysis.support_resistance)
    candidates.extend(e.price for e in m15_analysis.equal_levels)
    candidates.extend(s.price for s in m15_analysis.swings)
    if direction == "long":
        return sorted(p for p in candidates if p > entry)
    return sorted((p for p in candidates if p < entry), reverse=True)


def conviction_multiplier(dominant_reason_pct: float, confirmation_pct: float) -> float:
    raw = (
        1.0
        + 0.5 * (dominant_reason_pct - DOMINANT_REASON_THRESHOLD) / 20.0
        + 0.5 * (confirmation_pct - CONFIRMATION_THRESHOLD) / 30.0
    )
    return max(CONVICTION_MIN, min(CONVICTION_MAX, raw))


def _price_to_r(direction: str, entry: float, original_sl: float, price: float) -> float:
    risk = (entry - original_sl) if direction == "long" else (original_sl - entry)
    if risk <= 0:
        return 0.0
    moved = (price - entry) if direction == "long" else (entry - price)
    return moved / risk


def simulate_managed_trade(
    *,
    symbol: str,
    direction: str,
    regime: str,
    dominant_reason_kind: str,
    signal_bar_idx: int,
    m15_df: pd.DataFrame,
    m15_analysis: TimeframeAnalysis,
    original_sl: float,
    partial_target: float | None,
    final_target: float,
    symbol_cost: SymbolCost,
    base_risk_pct: float,
    conviction_mult: float,
    timeframe: str = "M15",
) -> ManagedTradeResult | None:
    fill_idx = signal_bar_idx + 1
    if fill_idx >= len(m15_df):
        return None

    fill_price = float(m15_df["open"].iloc[fill_idx])
    entry_time = m15_df["timestamp"].iloc[fill_idx]

    risk_price = (fill_price - original_sl) if direction == "long" else (original_sl - fill_price)
    reward_price = (final_target - fill_price) if direction == "long" else (fill_price - final_target)
    if risk_price <= 0 or reward_price <= 0:
        return None  # gap between signal and fill flipped the setup invalid

    cost_r = cost_in_r(symbol_cost, risk_price)
    trade_side = "bullish" if direction == "long" else "bearish"
    opposite_side = "bearish" if direction == "long" else "bullish"
    deadline = entry_time + MAX_HOLDING

    current_sl = original_sl
    partial_taken = False
    partial_r = 0.0
    sl_ratchets = 0
    applied_bos_ids: set[int] = set()

    last_idx = fill_idx
    for k in range(fill_idx, len(m15_df)):
        bar_open_ts = m15_df["timestamp"].iloc[k]
        if bar_open_ts > deadline:
            break
        last_idx = k
        bar_high = float(m15_df["high"].iloc[k])
        bar_low = float(m15_df["low"].iloc[k])
        bar_close_ts = bar_open_ts + TF_DURATION[timeframe]

        favorable_bos = None
        opposing_bos = False
        for ev in m15_analysis.structure_events:
            if not (entry_time < ev.timestamp <= bar_close_ts):
                continue
            if ev.direction == trade_side and id(ev) not in applied_bos_ids:
                applied_bos_ids.add(id(ev))
                favorable_bos = ev
            elif ev.direction == opposite_side:
                opposing_bos = True

        if favorable_bos is not None:
            broken_level = favorable_bos.meta.get("broken_level")
            if broken_level is not None:
                buffer = structural_buffer(m15_analysis.df)
                candidate_sl = broken_level - buffer if direction == "long" else broken_level + buffer
                improves = candidate_sl > current_sl if direction == "long" else candidate_sl < current_sl
                if improves:
                    current_sl = candidate_sl
                    sl_ratchets += 1

        # Convencion conservadora (heredada de simulate_trade): si el stop y
        # un objetivo tambien fueron tocados en la misma vela, se asume que
        # el stop se activo primero -- solo OHLC disponible, no datos de tick.
        sl_hit = (bar_low <= current_sl) if direction == "long" else (bar_high >= current_sl)
        if sl_hit:
            if partial_taken:
                remainder_r = _price_to_r(direction, fill_price, original_sl, current_sl)
                r = 0.5 * partial_r + 0.5 * remainder_r
                reason = "trailing_stop_post_partial"
            else:
                r = _price_to_r(direction, fill_price, original_sl, current_sl)
                reason = "trailing_stop" if sl_ratchets > 0 else "stop_loss"
            return _finish(
                symbol, direction, regime, dominant_reason_kind, entry_time, fill_price, original_sl,
                partial_target, final_target, partial_taken, sl_ratchets, current_sl, bar_close_ts, k, reason, r,
                cost_r, conviction_mult, base_risk_pct,
            )

        if opposing_bos:
            exit_price = float(m15_df["close"].iloc[k])
            remainder_r = _price_to_r(direction, fill_price, original_sl, exit_price)
            r = (0.5 * partial_r + 0.5 * remainder_r) if partial_taken else remainder_r
            reason = "structural_invalidation_post_partial" if partial_taken else "structural_invalidation_pre_partial"
            return _finish(
                symbol, direction, regime, dominant_reason_kind, entry_time, fill_price, original_sl,
                partial_target, final_target, partial_taken, sl_ratchets, current_sl, bar_close_ts, k, reason, r,
                cost_r, conviction_mult, base_risk_pct,
            )

        if not partial_taken and partial_target is not None:
            partial_hit = (bar_high >= partial_target) if direction == "long" else (bar_low <= partial_target)
            if partial_hit:
                partial_taken = True
                partial_r = _price_to_r(direction, fill_price, original_sl, partial_target)
                breakeven_sl = fill_price
                current_sl = max(current_sl, breakeven_sl) if direction == "long" else min(current_sl, breakeven_sl)
                continue

        tp_hit = (bar_high >= final_target) if direction == "long" else (bar_low <= final_target)
        if tp_hit:
            tp_r = _price_to_r(direction, fill_price, original_sl, final_target)
            r = (0.5 * partial_r + 0.5 * tp_r) if partial_taken else tp_r
            reason = "take_profit_post_partial" if partial_taken else "take_profit"
            return _finish(
                symbol, direction, regime, dominant_reason_kind, entry_time, fill_price, original_sl,
                partial_target, final_target, partial_taken, sl_ratchets, current_sl, bar_close_ts, k, reason, r,
                cost_r, conviction_mult, base_risk_pct,
            )

    exit_price = float(m15_df["close"].iloc[last_idx])
    exit_ts = m15_df["timestamp"].iloc[last_idx] + TF_DURATION[timeframe]
    tail_r = _price_to_r(direction, fill_price, original_sl, exit_price)
    r = (0.5 * partial_r + 0.5 * tail_r) if partial_taken else tail_r
    return _finish(
        symbol, direction, regime, dominant_reason_kind, entry_time, fill_price, original_sl,
        partial_target, final_target, partial_taken, sl_ratchets, current_sl, exit_ts, last_idx,
        "max_holding_period", r, cost_r, conviction_mult, base_risk_pct,
    )


def _finish(
    symbol, direction, regime, dominant_reason_kind, entry_time, fill_price, original_sl,
    partial_target, final_target, partial_taken, sl_ratchets, final_sl, exit_time, exit_bar_index, exit_reason,
    gross_r, cost_r, conviction_mult, base_risk_pct,
) -> ManagedTradeResult:
    return ManagedTradeResult(
        symbol=symbol,
        direction=direction,
        regime=regime,
        dominant_reason_kind=dominant_reason_kind,
        entry_time=entry_time,
        entry=fill_price,
        original_sl=original_sl,
        final_sl=final_sl,
        partial_target=partial_target,
        final_target=final_target,
        partial_taken=partial_taken,
        sl_ratchets=sl_ratchets,
        exit_time=exit_time,
        exit_bar_index=exit_bar_index,
        exit_reason=exit_reason,
        gross_r=gross_r,
        cost_r=cost_r,
        net_r=gross_r - cost_r,
        conviction_multiplier=conviction_mult,
        risk_pct_applied=base_risk_pct * conviction_mult,
    )
