"""Runs the real `MultiTimeframePipeline` over years of history for one
symbol. Detectors run ONCE per timeframe over the full dataset
(`TimeframeAnalysis.from_candles`); each candidate M15 confirmation bar is
then replayed cheaply via `.as_of()` -- see pipeline/engine.py's module
docstring for why that's guaranteed equivalent to recomputing from scratch.

Only M15 bars that fall within `config.m15_confirmation.window_candles` of a
confirmation-shaped event are used as anchors -- a signal requires a
confirmation within that same window (see `_latest_confirmation`), so every
other bar could not possibly produce one. With `window_candles=1` (the
original strict behavior) this is exactly "the bar the event itself landed
on"; with `window_candles>1` it also includes the `window_candles-1` bars
that follow each event, since those are now valid anchors too. This turns
"scan every bar of years of M15" (tens of thousands of full pipeline
evaluations) into "scan every bar within reach of something confirmation-
shaped" (still a small fraction), without changing what the backtest finds
-- as long as this window matches `_latest_confirmation`'s, which is the
entire point of both reading `config.m15_confirmation.window_candles` rather
than each hardcoding their own value (see test_backtest_engine.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trader.backtest.costs import SymbolCost
from trader.backtest.trade import TradeResult, simulate_trade
from trader.config import TraderConfig
from trader.events import TF_DURATION
from trader.pipeline.confluence import FAMILY_BY_KIND
from trader.pipeline.engine import MultiTimeframePipeline, NoSignal, TimeframeAnalysis
from trader.signal import TradingSignal

_CONFIRMATION_KINDS = {"choch", "turtle_soup_bullish", "turtle_soup_bearish", "sharp_turn"}


@dataclass(frozen=True)
class NoSignalRecord:
    symbol: str
    as_of: pd.Timestamp
    stage: str
    reason: str


@dataclass(frozen=True)
class SymbolBacktestResult:
    symbol: str
    trades: list[TradeResult]
    no_signals: list[NoSignalRecord]


def _candidate_confirmation_timestamps(m15_analysis: TimeframeAnalysis, window_candles: int) -> list[pd.Timestamp]:
    # Uses `m15_analysis.timeframe` rather than a literal "M15" -- see the
    # matching note on pipeline/engine.py's `_latest_confirmation`, which this
    # must stay synchronized with (same reasoning: every production caller
    # passes M15 today, identical behavior either way).
    step = TF_DURATION[m15_analysis.timeframe]
    timestamps: set[pd.Timestamp] = set()
    for e in m15_analysis.all_events():
        if e.kind in _CONFIRMATION_KINDS or e.kind.startswith("inverted_fair_value_gap"):
            for k in range(window_candles):
                timestamps.add(e.timestamp + k * step)
    return sorted(timestamps)


def _dominant_confluence_family(signal: TradingSignal) -> str:
    return FAMILY_BY_KIND.get(signal.confirmation.kind, "unknown")


def run_symbol_backtest(
    symbol: str,
    full_candles: dict[str, pd.DataFrame],
    config: TraderConfig,
    symbol_cost: SymbolCost,
) -> SymbolBacktestResult:
    full_analyses = {tf: TimeframeAnalysis.from_candles(full_candles[tf], tf, config) for tf in ("D1", "H1", "M15")}
    return run_symbol_backtest_from_analyses(symbol, full_analyses, full_candles["M15"], config, symbol_cost)


def run_symbol_backtest_from_analyses(
    symbol: str,
    full_analyses: dict[str, TimeframeAnalysis],
    m15_df: pd.DataFrame,
    config: TraderConfig,
    symbol_cost: SymbolCost,
) -> SymbolBacktestResult:
    """Same as `run_symbol_backtest`, but takes already-computed detector
    output instead of recomputing it. Valid whenever `config` only varies
    fields that affect DECISION logic (poi/confluence/sessions/risk), not
    detector computation (structure/liquidity/fvg/support_resistance/elliott)
    -- exactly the case for a parameter sweep over the former, where
    recomputing detectors identically for every combination would be pure
    waste.
    """
    m15_close_ts = m15_df["timestamp"] + TF_DURATION["M15"]
    idx_by_close_ts = {t: i for i, t in enumerate(m15_close_ts)}

    pipeline = MultiTimeframePipeline(config)
    candidates = _candidate_confirmation_timestamps(full_analyses["M15"], config.m15_confirmation.window_candles)

    trades: list[TradeResult] = []
    no_signals: list[NoSignalRecord] = []
    busy_until_idx = -1

    for as_of in candidates:
        bar_idx = idx_by_close_ts.get(as_of)
        if bar_idx is None or bar_idx <= busy_until_idx:
            continue

        current_price = float(m15_df["close"].iloc[bar_idx])
        analyses_as_of = {tf: full_analyses[tf].as_of(as_of) for tf in ("D1", "H1", "M15")}
        result = pipeline.run_from_analyses(symbol, analyses_as_of, as_of, current_price)

        if isinstance(result, NoSignal):
            no_signals.append(NoSignalRecord(symbol=symbol, as_of=as_of, stage=result.stage, reason=result.reason))
            continue

        trade = simulate_trade(
            result, m15_df, bar_idx, full_analyses["H1"], symbol_cost, _dominant_confluence_family(result)
        )
        if trade is None:
            no_signals.append(NoSignalRecord(symbol=symbol, as_of=as_of, stage="fill", reason="invalidated_at_fill"))
            continue

        trades.append(trade)
        busy_until_idx = trade.exit_bar_index

    return SymbolBacktestResult(symbol=symbol, trades=trades, no_signals=no_signals)
