import pandas as pd

from trader.detectors.support_resistance import detect_support_resistance
from trader.events import MarketEvent


def test_level_confirmed_at_second_touch_timestamp_not_first():
    base_ts = pd.Timestamp("2026-01-05", tz="UTC")
    touch1 = MarketEvent.point("swing_high", "H1", base_ts, "neutral", 100.0)
    touch2 = MarketEvent.point("swing_low", "H1", base_ts + pd.Timedelta(hours=5), "neutral", 100.02)
    unrelated = MarketEvent.point("swing_high", "H1", base_ts + pd.Timedelta(hours=1), "neutral", 150.0)

    events = detect_support_resistance([touch1, touch2, unrelated], "H1", tolerance_pct=0.05, min_touches=2)

    assert len(events) == 1
    e = events[0]
    assert e.kind == "support_resistance"
    assert e.meta["touches"] == 2
    assert e.timestamp == base_ts + pd.Timedelta(hours=5)


def test_single_touch_below_min_touches_yields_nothing():
    base_ts = pd.Timestamp("2026-01-05", tz="UTC")
    touch1 = MarketEvent.point("swing_high", "H1", base_ts, "neutral", 100.0)

    events = detect_support_resistance([touch1], "H1", tolerance_pct=0.05, min_touches=2)
    assert events == []
