"""Experimento aislado: D1 (bias) -> un solo timeframe bajo (M15 o M30) para
POI + confirmacion + ejecucion, per prompt-experimento-d1-30m-15m.md.

NO modifica `MultiTimeframePipeline` (produccion) -- pipeline paralelo que
reusa los detectores ya validados (structure/liquidity/order_blocks/fvg/
common) via el mismo `TimeframeAnalysis.from_candles`, solo con
timeframe="M30" (o "M15") y umbrales re-derivados. Ver
logs/experiment_d1_ltf_thresholds.md para la justificacion completa de cada
umbral.

D1 (bias, categoria, disparador de reversion) sin cambios -- reusa
`TimeframeAnalysis.bias()` y `MultiTimeframePipeline._d1_reversal_trigger`
tal cual. Gates duros sin cambios: R:R>=1:2, piso de SL por ATR, filtro de
noticias. Sistema de scoring sin cambios (min_score_trend/reversal de
config.yaml) -- este experimento es sobre la arquitectura de timeframes, no
sobre revisar otra vez la capa de decision.

Diseno explicito sin H1: `simulate_trade`'s chequeo de invalidacion HTF
durante la operacion usa D1 (la unica temporalidad superior que queda),
pasando `d1_analysis` donde production pasa `h1_analysis` -- funciona sin
cambios porque esa funcion solo lee `.structure_events` de lo que se le
pasa.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trader.backtest.costs import SymbolCost
from trader.backtest.engine import NoSignalRecord, SymbolBacktestResult, _CONFIRMATION_KINDS, _dominant_confluence_family
from trader.backtest.trade import TradeResult, simulate_trade
from trader.config import TraderConfig
from trader.events import TF_DURATION, MarketEvent
from trader.news_filter import is_high_impact_news_window
from trader.pipeline.confluence import evaluate_confluences
from trader.pipeline.engine import MultiTimeframePipeline, NoSignal, TimeframeAnalysis
from trader.pipeline.scoring import compute_score
from trader.risk.levels import compute_trade_levels
from trader.sessions import classify_session
from trader.signal import TradingSignal

_STAGE_ORDER = ("D1_bias", "session", "news_filter", "LTF_poi", "LTF_confirmation", "score", "risk")


@dataclass(frozen=True)
class LtfThresholds:
    timeframe: str  # "M15" | "M30"
    grace_candles: int  # invalidation grace period, in this timeframe's own candles
    window_candles: int  # confirmation freshness window, in this timeframe's own candles
    structural_buffer_lookback: int  # candles for the recent-range SL buffer
    fvg_min_gap_pct: float  # minimum FVG width, see prompt-fix-fvg-minimo-ltf.md


# See logs/experiment_d1_ltf_thresholds.md for the full derivation of grace/window/buffer.
# fvg_min_gap_pct: 2x the widest of the 3 symbols' real observed M15/M30
# spread (GBPUSD, 0.0057%/0.0053%) -- see logs/fvg_min_gap_derivation.md. A
# gap only as wide as the spread would be entirely consumed by the cost of
# trading it; 2x leaves genuine room beyond pure transaction cost without
# the extra, less-justified aggressiveness of 3x (which has no evidence
# behind it beyond "more filtering").
LTF_15M = LtfThresholds(timeframe="M15", grace_candles=6, window_candles=3, structural_buffer_lookback=80, fvg_min_gap_pct=0.0114)
LTF_30M = LtfThresholds(timeframe="M30", grace_candles=3, window_candles=2, structural_buffer_lookback=40, fvg_min_gap_pct=0.0105)


def _structural_buffer(ltf_df: pd.DataFrame, lookback: int) -> float:
    if len(ltf_df) == 0:
        return 0.0001
    recent_range = (ltf_df["high"] - ltf_df["low"]).tail(lookback).mean()
    return float(recent_range) * 0.1


def _next_liquidity_target(trade_direction: str, ltf_analysis: TimeframeAnalysis, entry: float) -> float | None:
    candidates: list[float] = []
    candidates.extend(e.price for e in ltf_analysis.support_resistance)
    candidates.extend(e.price for e in ltf_analysis.equal_levels)
    candidates.extend(e.price for e in ltf_analysis.swings)
    if trade_direction == "long":
        above = [p for p in candidates if p > entry]
        return min(above) if above else None
    below = [p for p in candidates if p < entry]
    return max(below) if below else None


def build_ltf_analysis(df: pd.DataFrame, thresholds: LtfThresholds, config: TraderConfig) -> TimeframeAnalysis:
    """Reuses TimeframeAnalysis.from_candles directly (same detector suite,
    same code path production uses), overriding two things per-variant:
    the zone invalidation grace period (this timeframe's own candles, not
    the default M15-based one) and the minimum FVG gap size (see
    prompt-fix-fvg-minimo-ltf.md -- `config.fvg.min_gap_pct=0.0` was
    calibrated for H1's naturally large gaps; unchanged at M15/M30 it let
    spread-sized noise through as "zones"). OB thresholds
    (`structure.displacement_atr_multiple`) are untouched, per that same
    prompt -- the audit found no size problem there.
    """
    from dataclasses import replace

    grace_period = TF_DURATION[thresholds.timeframe] * thresholds.grace_candles
    variant_config = replace(config, fvg=replace(config.fvg, min_gap_pct=thresholds.fvg_min_gap_pct))
    return TimeframeAnalysis.from_candles(df, thresholds.timeframe, variant_config, grace_period=grace_period)


def evaluate_direction(
    symbol: str,
    direction: str,
    category: str,
    min_score: float,
    d1_analysis: TimeframeAnalysis,
    ltf_analysis: TimeframeAnalysis,
    thresholds: LtfThresholds,
    config: TraderConfig,
    session,
    as_of: pd.Timestamp,
    current_price: float,
    bias: str,
) -> TradingSignal | NoSignal:
    tol = config.poi.tolerance_pct / 100.0
    poi_candidates = [
        z
        for z in (ltf_analysis.order_blocks + ltf_analysis.fvgs)
        if z.direction == direction
        and z.confirmed_at is not None
        and z.confirmed_at <= as_of
        and (z.broken_at is None or as_of < z.broken_at)
        and z.overlaps(current_price * (1 - tol), current_price * (1 + tol))
    ]
    if not poi_candidates:
        return NoSignal(reason="no_ltf_poi_at_price", stage="LTF_poi")
    poi = max(poi_candidates, key=lambda z: z.timestamp)

    confirmation = MultiTimeframePipeline._latest_confirmation(ltf_analysis, direction, thresholds.window_candles)
    if confirmation is None:
        return NoSignal(reason="no_ltf_confirmation", stage="LTF_confirmation")

    pool = d1_analysis.all_events() + ltf_analysis.all_events()
    if session.overlap_london_ny or session.killzone_london or session.killzone_new_york:
        pool = pool + [MarketEvent.point("session", thresholds.timeframe, as_of, direction, current_price)]
    relevant = [e for e in pool if e.direction in (direction, "neutral")]
    confluences = evaluate_confluences(relevant, min_confluences=0, min_timeframes=0)
    score = compute_score(
        direction, d1_analysis, ltf_analysis, ltf_analysis, poi, confirmation, session,
        confluences.families, confluences.timeframes, as_of,
    )
    if score.total < min_score:
        return NoSignal(reason="score_below_minimum", stage="score")

    trade_direction = "long" if direction == "bullish" else "short"
    buffer = _structural_buffer(ltf_analysis.df, thresholds.structural_buffer_lookback)
    structural_stop = poi.price_low - buffer if trade_direction == "long" else poi.price_high + buffer

    risk_price = (current_price - structural_stop) if trade_direction == "long" else (structural_stop - current_price)
    min_risk = None
    if len(ltf_analysis.atr) > 0 and not pd.isna(ltf_analysis.atr.iloc[-1]):
        min_risk = config.risk.min_risk_atr_multiple * float(ltf_analysis.atr.iloc[-1])
    if min_risk is not None and risk_price < min_risk:
        return NoSignal(reason="risk_below_minimum_floor", stage="risk")

    target = _next_liquidity_target(trade_direction, ltf_analysis, current_price)
    if target is None:
        return NoSignal(reason="no_valid_target", stage="risk")

    levels = compute_trade_levels(trade_direction, current_price, structural_stop, target, config.risk.min_risk_reward)
    if levels is None:
        return NoSignal(reason="risk_reward_below_minimum", stage="risk")

    return TradingSignal(
        symbol=symbol, direction=trade_direction, category=category, bias_1d=bias, session=session,
        poi=poi, confirmation=confirmation, confluences=confluences, score=score, levels=levels,
        partial_at_progress_pct=config.risk.partial_at_progress_pct, generated_at=as_of,
    )


def run_from_analyses(
    symbol: str,
    d1_analysis: TimeframeAnalysis,
    ltf_analysis: TimeframeAnalysis,
    thresholds: LtfThresholds,
    config: TraderConfig,
    as_of: pd.Timestamp,
    current_price: float,
) -> TradingSignal | NoSignal:
    if len(d1_analysis.df) < 10 or len(ltf_analysis.df) < 10:
        return NoSignal(reason="insufficient_history", stage="D1_bias")

    bias = d1_analysis.bias()
    if bias is None:
        return NoSignal(reason="no_d1_bias_established", stage="D1_bias")

    session = classify_session(as_of, config.sessions)
    session_ok = (
        (session.overlap_london_ny or session.killzone_london or session.killzone_new_york)
        if config.sessions.require_overlap_or_killzone
        else session.any_active
    )
    if not session_ok:
        return NoSignal(reason="outside_active_sessions", stage="session")

    blocked, _ = is_high_impact_news_window(as_of, config.news_filter)
    if blocked:
        return NoSignal(reason="high_impact_news_window", stage="news_filter")

    bias_direction = "bullish" if bias == "up" else "bearish"
    opposite_direction = "bearish" if bias_direction == "bullish" else "bullish"

    latest_d1_structure = max(d1_analysis.structure_events, key=lambda e: e.timestamp)
    primary_category = "reversal" if latest_d1_structure.kind == "choch" else "trend"
    primary_min_score = config.scoring.min_score_reversal if primary_category == "reversal" else config.scoring.min_score_trend
    attempts = [(bias_direction, primary_category, primary_min_score)]

    if MultiTimeframePipeline._d1_reversal_trigger(d1_analysis, opposite_direction) is not None:
        attempts.append((opposite_direction, "reversal", config.scoring.min_score_reversal))

    last_result: NoSignal | None = None
    for direction, category, min_score in attempts:
        result = evaluate_direction(
            symbol, direction, category, min_score, d1_analysis, ltf_analysis, thresholds, config,
            session, as_of, current_price, bias,
        )
        if isinstance(result, TradingSignal):
            return result
        last_result = result

    assert last_result is not None
    return last_result


def _candidate_confirmation_timestamps(ltf_analysis: TimeframeAnalysis, window_candles: int) -> list[pd.Timestamp]:
    step = TF_DURATION[ltf_analysis.timeframe]
    timestamps: set[pd.Timestamp] = set()
    for e in ltf_analysis.all_events():
        if e.kind in _CONFIRMATION_KINDS or e.kind.startswith("inverted_fair_value_gap"):
            for k in range(window_candles):
                timestamps.add(e.timestamp + k * step)
    return sorted(timestamps)


def run_symbol_backtest_ltf(
    symbol: str,
    d1_analysis: TimeframeAnalysis,
    ltf_analysis: TimeframeAnalysis,
    ltf_df: pd.DataFrame,
    thresholds: LtfThresholds,
    config: TraderConfig,
    symbol_cost: SymbolCost,
) -> SymbolBacktestResult:
    """D1->LTF-unico equivalent of backtest/engine.py's
    run_symbol_backtest_from_analyses -- same walk structure (candidate
    anchors = LTF bars within window_candles of a confirmation-shaped event,
    `.as_of()` replay, `busy_until_idx` to skip bars already inside an open
    trade), adapted to a single execution/zone timeframe and D1 (not H1) as
    the HTF-invalidation reference for `simulate_trade`.
    """
    ltf_close_ts = ltf_df["timestamp"] + TF_DURATION[thresholds.timeframe]
    idx_by_close_ts = {t: i for i, t in enumerate(ltf_close_ts)}

    candidates = _candidate_confirmation_timestamps(ltf_analysis, thresholds.window_candles)

    trades: list[TradeResult] = []
    no_signals: list[NoSignalRecord] = []
    busy_until_idx = -1

    for as_of in candidates:
        bar_idx = idx_by_close_ts.get(as_of)
        if bar_idx is None or bar_idx <= busy_until_idx:
            continue

        current_price = float(ltf_df["close"].iloc[bar_idx])
        d1_asof = d1_analysis.as_of(as_of)
        ltf_asof = ltf_analysis.as_of(as_of)
        result = run_from_analyses(symbol, d1_asof, ltf_asof, thresholds, config, as_of, current_price)

        if isinstance(result, NoSignal):
            no_signals.append(NoSignalRecord(symbol=symbol, as_of=as_of, stage=result.stage, reason=result.reason))
            continue

        trade = simulate_trade(
            result, ltf_df, bar_idx, d1_analysis, symbol_cost, _dominant_confluence_family(result),
            timeframe=thresholds.timeframe,
        )
        if trade is None:
            no_signals.append(NoSignalRecord(symbol=symbol, as_of=as_of, stage="fill", reason="invalidated_at_fill"))
            continue

        trades.append(trade)
        busy_until_idx = trade.exit_bar_index

    return SymbolBacktestResult(symbol=symbol, trades=trades, no_signals=no_signals)
