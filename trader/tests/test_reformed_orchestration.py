"""Unit tests for trader/pipeline/reformed.py's own helpers (as opposed to
tests/test_reformed_pipeline.py's full end-to-end smoke test).

test_recent_sweep_requires_price_proximity is a regression test for a real
bug found while auditing the first full backtest run of the reformulated
system: n=8683 trades, ~1000x every prior baseline in this project's
history. The cause was `_recent_sweep` admitting any same-direction sweep
from the last 20 days regardless of how far price had since moved away from
the swept level -- unlike `_overlapping_zone`/`_nearby_level`, which both
correctly require the current price to be near the reference. Empirically,
on real EURUSD data, the swept level's price was a median of 0.22% (up to
2.86%) away from the current price when this fired -- nothing like "the
reason price is moving from here" (Capa 2 of the design doc).
"""

import pandas as pd

from tests.conftest import build_candles
from trader.events import MarketEvent
from trader.pipeline.engine import TimeframeAnalysis
from trader.pipeline.reformed import SWEEP_RECENCY_WINDOW, _recent_sweep, build_dominant_reason_engine


def test_recent_sweep_requires_price_proximity():
    as_of = pd.Timestamp("2026-01-20", tz="UTC")
    far_sweep = MarketEvent.point("liquidity_sweep_bullish", "H1", as_of - pd.Timedelta(days=1), "bullish", 90.0)
    close_sweep = MarketEvent.point("liquidity_sweep_bullish", "H1", as_of - pd.Timedelta(days=2), "bullish", 100.05)

    # far_sweep is more RECENT but geometrically irrelevant (price is nowhere near 90);
    # close_sweep is older but still close to current price -- must be the one returned.
    result = _recent_sweep([far_sweep, close_sweep], "bullish", as_of, current_price=100.0, tol=0.01)
    assert result is close_sweep


def test_recent_sweep_returns_none_when_nothing_is_close():
    as_of = pd.Timestamp("2026-01-20", tz="UTC")
    far_sweep = MarketEvent.point("liquidity_sweep_bullish", "H1", as_of - pd.Timedelta(days=1), "bullish", 90.0)
    assert _recent_sweep([far_sweep], "bullish", as_of, current_price=100.0, tol=0.01) is None


def test_recent_sweep_respects_recency_window_even_if_close():
    as_of = pd.Timestamp("2026-01-20", tz="UTC")
    old_but_close = MarketEvent.point(
        "liquidity_sweep_bullish", "H1", as_of - SWEEP_RECENCY_WINDOW - pd.Timedelta(days=1), "bullish", 100.0
    )
    assert _recent_sweep([old_but_close], "bullish", as_of, current_price=100.0, tol=0.01) is None


def test_recent_sweep_ignores_wrong_direction():
    as_of = pd.Timestamp("2026-01-20", tz="UTC")
    wrong_direction = MarketEvent.point("liquidity_sweep_bearish", "H1", as_of - pd.Timedelta(days=1), "bearish", 100.0)
    assert _recent_sweep([wrong_direction], "bullish", as_of, current_price=100.0, tol=0.01) is None


def _analysis(df, sweeps=None, timeframe="H1") -> TimeframeAnalysis:
    return TimeframeAnalysis(
        timeframe=timeframe, df=df, swings=[], structure_events=[], order_blocks=[], inverted_order_blocks=[],
        fvgs=[], inverted_fvgs=[], sweeps=sweeps or [], equal_levels=[], turtle_soups=[], sharp_turns=[],
        support_resistance=[], elliott=[],
    )


def test_sweep_below_absolute_wick_floor_is_never_ingested():
    """Regression/coverage for prompt-3-chequeos-antes-de-cerrar.md, chequeo
    1: sweep_wick_min_pct (existing config) only requires the wick be a
    minimum PROPORTION of its own candle's range -- a tiny candle with a
    proportionally clean wick still had no absolute-size floor at all."""
    df = build_candles([(10.0, 12.0, 9.0, 10.0)], freq="1h")  # range=3, bearish wick=2 (absolute)
    sweep = MarketEvent.point("liquidity_sweep_bearish", "H1", df["timestamp"].iloc[0], "bearish", 11.5)
    h1_analysis = _analysis(df, sweeps=[sweep])
    d1_analysis = _analysis(build_candles([(10.0, 10.0, 10.0, 10.0)], freq="1D"), timeframe="D1")

    engine_no_floor = build_dominant_reason_engine(d1_analysis, h1_analysis, min_sweep_wick_price=0.0)
    assert len(engine_no_floor._liquidity_tracker["bearish"]) == 1

    engine_below_wick = build_dominant_reason_engine(d1_analysis, h1_analysis, min_sweep_wick_price=1.0)
    assert len(engine_below_wick._liquidity_tracker["bearish"]) == 1  # 1.0 < absolute wick of 2.0 -> still ingested

    engine_above_wick = build_dominant_reason_engine(d1_analysis, h1_analysis, min_sweep_wick_price=3.0)
    assert len(engine_above_wick._liquidity_tracker["bearish"]) == 0  # 3.0 > absolute wick of 2.0 -> excluded
