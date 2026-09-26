"""Forward-simulates ONE trade from a `TradingSignal` bar-by-bar on M15 OHLC.

This is deliberately NOT a second signal engine -- the signal itself (POI,
confirmation, confluence, R:R) is always produced by
`MultiTimeframePipeline.run_from_analyses`, the exact same code the live path
uses. What's here is purely position MANAGEMENT simulation, which live
trading doesn't need in this form (a live poll checks one current price at a
time; a backtest has to resolve exactly how a whole bar's high/low range
would have played out, which is a different, bar-aware problem).

Fill happens at the NEXT bar's open after the signal bar closes -- never the
signal bar's own close, which would use the same price to both trigger and
enter (look-ahead-adjacent, and a real order can't fill before the decision
to place it exists). Within a bar, if both a stop and a target/partial level
are technically touched, the stop is assumed to have hit first -- the
standard conservative convention when only OHLC (not tick data) is
available.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trader.backtest.costs import SymbolCost, cost_in_r
from trader.events import TF_DURATION
from trader.pipeline.engine import TimeframeAnalysis
from trader.signal import TradingSignal

MAX_HOLDING = pd.Timedelta(days=3)


@dataclass(frozen=True)
class TradeResult:
    symbol: str
    direction: str
    category: str  # "trend" | "reversal"
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry: float
    sl: float
    tp: float
    planned_rr: float
    actual_rr: float
    partial_taken: bool
    exit_time: pd.Timestamp
    exit_bar_index: int
    exit_reason: str
    gross_r: float
    cost_r: float
    net_r: float
    session_labels: list[str]
    confirmation_kind: str
    dominant_confluence_family: str
    confluence_families: list[str] = field(default_factory=list)
    score_total: float = 0.0
    score_breakdown: dict = field(default_factory=dict)


def _price_to_r(direction: str, entry: float, original_sl: float, price: float) -> float:
    risk = (entry - original_sl) if direction == "long" else (original_sl - entry)
    if risk <= 0:
        return 0.0
    moved = (price - entry) if direction == "long" else (entry - price)
    return moved / risk


def simulate_trade(
    signal: TradingSignal,
    m15_df: pd.DataFrame,
    signal_bar_idx: int,
    h1_analysis: TimeframeAnalysis,
    symbol_cost: SymbolCost,
    dominant_confluence_family: str,
    timeframe: str = "M15",
) -> TradeResult | None:
    # `timeframe` names whatever timeframe `m15_df` (and `signal_bar_idx`)
    # actually are -- always M15 for every existing caller (the parameter
    # default preserves that exactly), but prompt-experimento-d1-30m-15m.md's
    # single-LTF pipeline executes on M30 too, where a hardcoded M15 bar-
    # close offset would silently corrupt entry_time/exit_time. `h1_analysis`
    # keeps its name for the same reason -- every existing caller passes H1
    # for HTF-invalidation-during-the-trade; the D1-LTF experiment passes D1
    # instead (the only higher timeframe left above a single execution tier),
    # which works unchanged since this function only ever reads
    # `.structure_events` off of whatever it's given.
    direction = signal.levels.direction
    original_sl = signal.levels.sl
    tp = signal.levels.tp
    fill_idx = signal_bar_idx + 1
    if fill_idx >= len(m15_df):
        return None

    fill_price = float(m15_df["open"].iloc[fill_idx])
    entry_time = m15_df["timestamp"].iloc[fill_idx]

    risk_price = (fill_price - original_sl) if direction == "long" else (original_sl - fill_price)
    reward_price = (tp - fill_price) if direction == "long" else (fill_price - tp)
    if risk_price <= 0 or reward_price <= 0:
        return None  # the gap between signal and fill flipped the setup invalid -- no trade
    actual_rr = reward_price / risk_price
    cost_r = cost_in_r(symbol_cost, risk_price)

    partial_price = fill_price + (tp - fill_price) * signal.partial_at_progress_pct

    opposite_direction = "bearish" if direction == "long" else "bullish"
    deadline = entry_time + MAX_HOLDING

    current_sl = original_sl
    partial_taken = False
    partial_r = 0.0

    last_idx = fill_idx
    for k in range(fill_idx, len(m15_df)):
        bar_open_ts = m15_df["timestamp"].iloc[k]
        if bar_open_ts > deadline:
            break
        last_idx = k
        bar_high = float(m15_df["high"].iloc[k])
        bar_low = float(m15_df["low"].iloc[k])
        bar_close_ts = bar_open_ts + TF_DURATION[timeframe]

        invalidated = any(
            e.kind == "choch" and e.direction == opposite_direction and entry_time < e.timestamp <= bar_close_ts
            for e in h1_analysis.structure_events
        )

        if not partial_taken:
            sl_hit = (bar_low <= current_sl) if direction == "long" else (bar_high >= current_sl)
            if sl_hit:
                return _finish(
                    signal, signal.symbol, direction, entry_time, fill_price, original_sl, tp, actual_rr, False,
                    bar_close_ts, k, "stop_loss", -1.0, cost_r, dominant_confluence_family,
                )

            if invalidated:
                exit_price = float(m15_df["close"].iloc[k])
                r = _price_to_r(direction, fill_price, original_sl, exit_price)
                return _finish(
                    signal, signal.symbol, direction, entry_time, fill_price, original_sl, tp, actual_rr, False,
                    bar_close_ts, k, "htf_invalidation_pre_partial", r, cost_r, dominant_confluence_family,
                )

            partial_hit = (bar_high >= partial_price) if direction == "long" else (bar_low <= partial_price)
            if partial_hit:
                partial_taken = True
                current_sl = fill_price
                partial_r = _price_to_r(direction, fill_price, original_sl, partial_price)
                continue

        else:
            be_hit = (bar_low <= current_sl) if direction == "long" else (bar_high >= current_sl)
            if be_hit:
                r = 0.5 * partial_r + 0.5 * 0.0
                return _finish(
                    signal, signal.symbol, direction, entry_time, fill_price, original_sl, tp, actual_rr, True,
                    bar_close_ts, k, "breakeven_stop", r, cost_r, dominant_confluence_family,
                )

            if invalidated:
                r = 0.5 * partial_r + 0.5 * 0.0
                return _finish(
                    signal, signal.symbol, direction, entry_time, fill_price, original_sl, tp, actual_rr, True,
                    bar_close_ts, k, "htf_invalidation_post_partial", r, cost_r, dominant_confluence_family,
                )

            tp_hit = (bar_high >= tp) if direction == "long" else (bar_low <= tp)
            if tp_hit:
                tp_r = _price_to_r(direction, fill_price, original_sl, tp)
                r = 0.5 * partial_r + 0.5 * tp_r
                return _finish(
                    signal, signal.symbol, direction, entry_time, fill_price, original_sl, tp, actual_rr, True,
                    bar_close_ts, k, "take_profit", r, cost_r, dominant_confluence_family,
                )

    exit_price = float(m15_df["close"].iloc[last_idx])
    exit_ts = m15_df["timestamp"].iloc[last_idx] + TF_DURATION[timeframe]
    tail_r = _price_to_r(direction, fill_price, original_sl, exit_price)
    r = (0.5 * partial_r + 0.5 * tail_r) if partial_taken else tail_r
    return _finish(
        signal, signal.symbol, direction, entry_time, fill_price, original_sl, tp, actual_rr, partial_taken,
        exit_ts, last_idx, "max_holding_period", r, cost_r, dominant_confluence_family,
    )


def _finish(
    signal: TradingSignal,
    symbol: str,
    direction: str,
    entry_time: pd.Timestamp,
    fill_price: float,
    original_sl: float,
    tp: float,
    actual_rr: float,
    partial_taken: bool,
    exit_time: pd.Timestamp,
    exit_bar_index: int,
    exit_reason: str,
    gross_r: float,
    cost_r: float,
    dominant_confluence_family: str,
) -> TradeResult:
    return TradeResult(
        symbol=symbol,
        direction=direction,
        category=signal.category,
        signal_time=signal.generated_at,
        entry_time=entry_time,
        entry=fill_price,
        sl=original_sl,
        tp=tp,
        planned_rr=signal.levels.risk_reward,
        actual_rr=actual_rr,
        partial_taken=partial_taken,
        exit_time=exit_time,
        exit_bar_index=exit_bar_index,
        exit_reason=exit_reason,
        gross_r=gross_r,
        cost_r=cost_r,
        net_r=gross_r - cost_r,
        session_labels=signal.session.active_labels,
        confirmation_kind=signal.confirmation.kind,
        dominant_confluence_family=dominant_confluence_family,
        confluence_families=sorted(signal.confluences.families),
        score_total=signal.score.total,
        score_breakdown=signal.score.as_dict(),
    )
