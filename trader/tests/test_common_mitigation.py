import pandas as pd

from trader.detectors.common import apply_mitigation
from trader.events import TF_DURATION, MarketEvent
from tests.conftest import build_candles

_H1 = TF_DURATION["H1"]
_GRACE = pd.Timedelta(hours=2)


def _bullish_zone(df) -> MarketEvent:
    return MarketEvent.zone(
        "order_block_bullish", "H1", df["timestamp"].iloc[0] + _H1, "bullish", price_high=102.0, price_low=100.0
    )


def test_zone_never_touched_is_confirmed_from_formation():
    bars = [(100.0, 100.5, 99.5, 100.0), (150.0, 150.5, 149.5, 150.0)]  # far away, never overlaps
    df = build_candles(bars, freq="1h")
    zone = _bullish_zone(df)

    zones, inversions = apply_mitigation([zone], df, "H1", _GRACE)

    assert zones[0].mitigated is False
    assert zones[0].mitigated_at is None
    assert zones[0].confirmed_at == zone.timestamp
    assert zones[0].broken_at is None
    assert inversions == []


def test_zone_touched_and_never_breaks_confirms_after_grace_period():
    bars = [
        (100.0, 100.5, 99.5, 100.0),  # formation
        (99.8, 101.0, 99.0, 100.5),  # touches, closes back inside -- respected
        (100.5, 100.6, 100.4, 100.5),  # continues, never breaks
    ]
    df = build_candles(bars, freq="1h")
    zone = _bullish_zone(df)

    zones, inversions = apply_mitigation([zone], df, "H1", _GRACE)

    touched_at = df["timestamp"].iloc[1] + _H1
    assert zones[0].mitigated is True
    assert zones[0].mitigated_at == touched_at
    assert zones[0].confirmed_at == touched_at + _GRACE
    assert zones[0].broken_at is None
    assert inversions == []


def test_zone_broken_same_candle_as_touch_never_confirmed():
    bars = [
        (100.0, 100.5, 99.5, 100.0),  # formation
        (99.8, 101.0, 99.0, 99.5),  # touches AND closes below 100 in the same candle
    ]
    df = build_candles(bars, freq="1h")
    zone = _bullish_zone(df)

    zones, inversions = apply_mitigation([zone], df, "H1", _GRACE)

    break_ts = df["timestamp"].iloc[1] + _H1
    assert zones[0].confirmed_at is None  # held=0 < grace period -- never a valid POI
    assert zones[0].broken_at == break_ts
    assert len(inversions) == 1
    assert inversions[0].kind == "inverted_order_block_bullish"
    assert inversions[0].timestamp == break_ts


def test_zone_broken_within_grace_period_never_confirmed():
    bars = [
        (100.0, 100.5, 99.5, 100.0),  # formation
        (99.8, 101.0, 99.0, 100.5),  # touch, closes inside
        (100.5, 100.6, 99.0, 99.5),  # 1h after touch (< 2h grace) -- closes below 100
    ]
    df = build_candles(bars, freq="1h")
    zone = _bullish_zone(df)

    zones, inversions = apply_mitigation([zone], df, "H1", _GRACE)

    assert zones[0].confirmed_at is None  # held=1h < grace=2h
    assert zones[0].broken_at == df["timestamp"].iloc[2] + _H1
    assert len(inversions) == 1


def test_zone_broken_after_grace_period_confirms_then_eventually_breaks():
    """Also a regression test for a real bug: an earlier version of
    apply_mitigation stopped scanning at the first touch, so a break several
    candles later (as here, 3 hours after the touch) was never detected at
    all -- this specifically exercises that multi-candle gap.
    """
    bars = [
        (100.0, 100.5, 99.5, 100.0),  # formation
        (99.8, 101.0, 99.0, 100.5),  # touch, closes inside
        (100.5, 100.6, 100.4, 100.5),  # +1h post-touch, holding
        (100.5, 100.6, 100.4, 100.5),  # +2h post-touch, holding (grace period met here)
        (100.5, 100.6, 99.0, 99.5),  # +3h post-touch -- closes below 100
    ]
    df = build_candles(bars, freq="1h")
    zone = _bullish_zone(df)

    zones, inversions = apply_mitigation([zone], df, "H1", _GRACE)

    touched_at = df["timestamp"].iloc[1] + _H1
    broken_at = df["timestamp"].iloc[4] + _H1
    assert (broken_at - touched_at) == pd.Timedelta(hours=3)  # sanity: genuinely past the 2h grace period
    assert zones[0].confirmed_at == touched_at + _GRACE  # held long enough -> confirmed
    assert zones[0].broken_at == broken_at  # ...but still eventually invalidated
    assert len(inversions) == 1
    assert inversions[0].timestamp == broken_at
