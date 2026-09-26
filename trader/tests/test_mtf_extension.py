from __future__ import annotations

import pandas as pd

from trader.events import MarketEvent
from trader.mtf_strategies.extension import compute_extension, find_swing_origin, passes_extension_filter


def _sw(kind: str, price: float, hours: int) -> MarketEvent:
    return MarketEvent.point(
        kind=kind, timeframe="H4", timestamp=pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=4 * hours),
        direction="neutral", price=price,
    )


def test_find_swing_origin_for_a_high_extremity_is_the_prior_swing_low():
    swings = [_sw("swing_low", 90, 1), _sw("swing_high", 100, 2), _sw("swing_low", 95, 3), _sw("swing_high", 120, 4)]
    extremity = swings[3]  # the swing_high at 120
    origin = find_swing_origin(swings, extremity)
    assert origin is not None
    assert origin.kind == "swing_low"
    assert origin.price == 95  # the low IMMEDIATELY before the extremity, not the older one (90)


def test_find_swing_origin_for_a_low_extremity_is_the_prior_swing_high():
    swings = [_sw("swing_high", 110, 1), _sw("swing_low", 100, 2), _sw("swing_high", 105, 3), _sw("swing_low", 80, 4)]
    extremity = swings[3]  # swing_low at 80
    origin = find_swing_origin(swings, extremity)
    assert origin is not None
    assert origin.kind == "swing_high"
    assert origin.price == 105


def test_find_swing_origin_none_without_prior_opposite_swing():
    swings = [_sw("swing_high", 100, 1)]
    origin = find_swing_origin(swings, swings[0])
    assert origin is None


def test_compute_extension_known_ratio():
    # |120 - 100| / 10 = 2.0x ATR
    assert compute_extension(current_price=120, swing_origin_price=100, atr_4h=10) == 2.0


def test_compute_extension_none_with_zero_or_missing_atr():
    assert compute_extension(120, 100, atr_4h=0) is None
    assert compute_extension(120, 100, atr_4h=None) is None


def test_passes_extension_filter_at_exactly_threshold_fails_strict_greater_than():
    # regla 8: "Extension > 1xATR_4H" -- estrictamente mayor, no >=
    assert passes_extension_filter(current_price=110, swing_origin_price=100, atr_4h=10, threshold=1.0) is False
    assert passes_extension_filter(current_price=110.01, swing_origin_price=100, atr_4h=10, threshold=1.0) is True


def test_passes_extension_filter_below_threshold_fails():
    assert passes_extension_filter(current_price=105, swing_origin_price=100, atr_4h=10, threshold=1.0) is False
