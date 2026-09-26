import pandas as pd

from trader.events import MarketEvent
from trader.pipeline.confluence import evaluate_confluences


def _event(kind, tf, direction="bullish"):
    return MarketEvent.point(kind, tf, pd.Timestamp("2026-01-05", tz="UTC"), direction, 1.0)


def test_passes_with_three_families_two_timeframes():
    events = [
        _event("bos", "H1"),
        _event("order_block_bullish", "H1"),
        _event("fair_value_gap_bullish", "M15"),
    ]
    check = evaluate_confluences(events, min_confluences=3, min_timeframes=2)
    assert check.passed is True
    assert check.families == {"structure", "order_block", "fvg"}
    assert check.timeframes == {"H1", "M15"}


def test_fails_when_all_same_timeframe():
    events = [
        _event("bos", "H1"),
        _event("order_block_bullish", "H1"),
        _event("fair_value_gap_bullish", "H1"),
    ]
    check = evaluate_confluences(events, min_confluences=3, min_timeframes=2)
    assert check.passed is False


def test_fails_when_below_minimum_families():
    events = [_event("bos", "H1"), _event("bos", "M15")]
    check = evaluate_confluences(events, min_confluences=3, min_timeframes=2)
    assert check.passed is False


def test_unknown_kind_ignored_not_counted_as_family():
    events = [_event("something_unmapped", "H1")]
    check = evaluate_confluences(events, min_confluences=1, min_timeframes=1)
    assert check.families == set()
    assert check.passed is False
