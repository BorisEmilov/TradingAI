from __future__ import annotations

import pandas as pd

from trader.events import MarketEvent
from trader.mtf_strategies.bias import classify_htf_structure


def _sw(kind: str, price: float, hours: int) -> MarketEvent:
    return MarketEvent.point(
        kind=kind, timeframe="H4", timestamp=pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=4 * hours),
        direction="neutral", price=price,
    )


def test_hh_hl_hh_gives_bullish():
    # H0=100, L0=90, H1=110 (HH), L1=95 (HL), H2=120 (HH) -- close still above the HL
    swings = [_sw("swing_high", 100, 1), _sw("swing_low", 90, 2), _sw("swing_high", 110, 3), _sw("swing_low", 95, 4), _sw("swing_high", 120, 5)]
    assert classify_htf_structure(swings, latest_close=118) == "bullish"


def test_ll_lh_ll_gives_bearish():
    # L0=100, H0=110, L1=90 (LL), H1=105 (LH), L2=80 (LL) -- close still below the LH
    swings = [_sw("swing_low", 100, 1), _sw("swing_high", 110, 2), _sw("swing_low", 90, 3), _sw("swing_high", 105, 4), _sw("swing_low", 80, 5)]
    assert classify_htf_structure(swings, latest_close=100) == "bearish"


def test_broken_hl_invalidates_bullish_structure():
    swings = [_sw("swing_high", 100, 1), _sw("swing_low", 90, 2), _sw("swing_high", 110, 3), _sw("swing_low", 95, 4), _sw("swing_high", 120, 5)]
    # close fell back BELOW the HL (95) -- the "último HL relevante" got broken
    assert classify_htf_structure(swings, latest_close=94) == "none"


def test_non_ascending_highs_is_ambiguous_none():
    # H2 (102) < H1 (105) -- not a higher high, regla 1's "rupturas falsas" case
    swings = [_sw("swing_high", 100, 1), _sw("swing_low", 90, 2), _sw("swing_high", 105, 3), _sw("swing_low", 95, 4), _sw("swing_high", 102, 5)]
    assert classify_htf_structure(swings, latest_close=101) == "none"


def test_sequence_ending_in_a_low_is_not_yet_confirmed():
    # otherwise bullish-shaped (H ascending, L ascending) but the LAST point
    # is a low, not the confirming fresh high -- pattern isn't complete yet
    swings = [_sw("swing_low", 90, 1), _sw("swing_high", 100, 2), _sw("swing_low", 95, 3), _sw("swing_high", 110, 4), _sw("swing_low", 98, 5)]
    assert classify_htf_structure(swings, latest_close=105) == "none"


def test_fewer_than_five_swings_is_none():
    swings = [_sw("swing_high", 100, 1), _sw("swing_low", 90, 2), _sw("swing_high", 110, 3)]
    assert classify_htf_structure(swings, latest_close=105) == "none"


def test_no_swings_at_all_is_none():
    assert classify_htf_structure([], latest_close=100) == "none"


def test_consecutive_same_kind_run_collapses_to_its_extreme():
    # two swing_highs in a row (100 then 105, no real low between) should
    # collapse to the higher one (105) before reading HH-HL-HH
    swings = [
        _sw("swing_low", 90, 1), _sw("swing_high", 100, 2), _sw("swing_high", 105, 3),
        _sw("swing_low", 95, 4), _sw("swing_high", 115, 5), _sw("swing_low", 102, 6), _sw("swing_high", 120, 7),
    ]
    assert classify_htf_structure(swings, latest_close=118) == "bullish"
