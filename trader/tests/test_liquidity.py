import pandas as pd

from trader.detectors.liquidity import (
    detect_equal_levels,
    detect_liquidity_sweeps,
    detect_sharp_turns,
    detect_turtle_soup,
)
from trader.events import MarketEvent
from tests.conftest import build_candles


def test_liquidity_sweep_bearish_wick_pierce_close_back_inside():
    swing = MarketEvent.point("swing_high", "M15", pd.Timestamp("2026-01-05", tz="UTC"), "neutral", 10.0)
    bars = [
        (9.95, 10.2, 9.8, 9.9),  # wicks above 10.0, closes back below
    ]
    df = build_candles(bars, start="2026-01-05 01:00:00", freq="15min")

    events = detect_liquidity_sweeps(df, "M15", [swing], min_wick_pct=10.0)

    assert len(events) == 1
    e = events[0]
    assert e.kind == "liquidity_sweep_bearish"
    assert e.direction == "bearish"
    assert e.price == 10.0


def test_liquidity_sweep_requires_minimum_wick_fraction():
    swing = MarketEvent.point("swing_high", "M15", pd.Timestamp("2026-01-05", tz="UTC"), "neutral", 10.0)
    bars = [
        (9.9, 10.01, 9.8, 9.95),  # barely pierces, tiny wick
    ]
    df = build_candles(bars, start="2026-01-05 01:00:00", freq="15min")

    events = detect_liquidity_sweeps(df, "M15", [swing], min_wick_pct=50.0)
    assert events == []


def test_equal_highs_clusters_within_tolerance():
    base_ts = pd.Timestamp("2026-01-05", tz="UTC")
    highs = [
        MarketEvent.point("swing_high", "H1", base_ts, "neutral", 100.0),
        MarketEvent.point("swing_high", "H1", base_ts + pd.Timedelta(hours=5), "neutral", 100.03),
        MarketEvent.point("swing_high", "H1", base_ts + pd.Timedelta(hours=10), "neutral", 105.0),  # too far
    ]
    events = detect_equal_levels(highs, "H1", tolerance_pct=0.05)

    assert len(events) == 1
    assert events[0].kind == "equal_highs"
    assert events[0].meta["count"] == 2
    assert events[0].timestamp == base_ts + pd.Timedelta(hours=5)


def test_turtle_soup_bullish_false_breakdown():
    flat = [(10.0, 10.1, 9.9, 10.0)] * 5
    reversal = [(9.9, 10.05, 9.7, 9.95)]  # low(9.7) < window_low(9.9), close(9.95) back above
    df = build_candles(flat + reversal, freq="15min")

    events = detect_turtle_soup(df, "M15", lookback_bars=5, min_wick_pct=10.0)

    assert len(events) == 1
    assert events[0].kind == "turtle_soup_bullish"
    assert events[0].price == 9.9


def test_sharp_turn_opposite_displacement_candles():
    prelude = [(10.0, 10.05, 9.95, 10.0)] * 15  # warm up ATR at ~0.1
    strong_up = (10.0, 10.6, 9.95, 10.55)  # big bullish
    strong_down = (10.55, 10.6, 9.9, 9.95)  # big bearish, reverses immediately
    df = build_candles(prelude + [strong_up, strong_down], freq="15min")

    from trader.detectors.indicators import atr as atr_fn

    atr_series = atr_fn(df, period=14)
    events = detect_sharp_turns(df, "M15", atr_series, displacement_atr_multiple=1.5)

    assert len(events) == 1
    assert events[0].kind == "sharp_turn"
    assert events[0].direction == "bearish"
