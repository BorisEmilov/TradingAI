from trader.detectors.fvg import detect_fvg
from trader.events import TF_DURATION
from tests.conftest import build_candles


def test_bullish_fvg_confirmed_at_c3_close():
    bars = [
        (10.0, 10.2, 9.8, 10.0),  # C1: high=10.2
        (10.0, 11.0, 10.1, 10.9),  # C2: impulse
        (10.9, 11.2, 10.6, 11.1),  # C3: low=10.6 > C1.high=10.2 -> gap
    ]
    df = build_candles(bars, freq="15min")

    events = detect_fvg(df, "M15")

    assert len(events) == 1
    e = events[0]
    assert e.kind == "fair_value_gap_bullish"
    assert e.price_high == 10.6
    assert e.price_low == 10.2
    assert e.timestamp == df["timestamp"].iloc[2] + TF_DURATION["M15"]


def test_bearish_fvg():
    bars = [
        (11.0, 11.2, 10.9, 11.0),  # C1: low=10.9
        (11.0, 11.0, 10.0, 10.1),  # C2: impulse down
        (10.1, 10.5, 9.9, 10.0),  # C3: high=10.5 < C1.low=10.9 -> gap
    ]
    df = build_candles(bars, freq="15min")

    events = detect_fvg(df, "M15")

    assert len(events) == 1
    e = events[0]
    assert e.kind == "fair_value_gap_bearish"
    assert e.price_high == 10.9
    assert e.price_low == 10.5


def test_no_gap_no_event():
    bars = [
        (10.0, 10.2, 9.8, 10.0),
        (10.0, 10.5, 9.9, 10.3),
        (10.3, 10.4, 10.0, 10.2),  # overlaps C1 -- no gap
    ]
    df = build_candles(bars, freq="15min")
    assert detect_fvg(df, "M15") == []
