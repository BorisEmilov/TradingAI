from __future__ import annotations

import pandas as pd

from trader.config import load_config
from trader.mtf_strategies.liquidity_levels import (
    compute_asia_session_high_low,
    compute_liquidity_levels,
    detect_named_level_sweep,
)
from tests.conftest import build_candles


def test_pdh_pdl_is_the_last_closed_day_never_the_current_one():
    dates = pd.bdate_range("2024-01-01", periods=15, tz="UTC")  # 3 business weeks
    df = pd.DataFrame({"timestamp": dates, "open": 1.0, "high": range(100, 115), "low": range(50, 65), "close": 1.0})

    lv = compute_liquidity_levels(df)

    assert lv.pdh == 114.0  # last row's own high -- caller already truncated causally
    assert lv.pdl == 64.0


def test_pwh_pwl_is_the_week_before_the_one_containing_the_last_row():
    dates = pd.bdate_range("2024-01-01", periods=15, tz="UTC")
    df = pd.DataFrame({"timestamp": dates, "open": 1.0, "high": range(100, 115), "low": range(50, 65), "close": 1.0})

    lv = compute_liquidity_levels(df)

    # week 2 (Mon 2024-01-08 .. Fri 2024-01-12) = rows 5..9, high 105..109, low 55..59
    assert lv.pwh == 109.0
    assert lv.pwl == 55.0


def test_pwh_pwl_none_with_less_than_two_weeks_of_history():
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    df = pd.DataFrame({"timestamp": dates, "open": 1.0, "high": [10, 11, 12], "low": [1, 2, 3], "close": 1.0})

    lv = compute_liquidity_levels(df)

    assert lv.pwh is None
    assert lv.pwl is None


def test_empty_history_gives_all_none():
    df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
    lv = compute_liquidity_levels(df)
    assert lv == type(lv)(None, None, None, None)


def test_detect_named_level_sweep_bullish_wick_and_close_back_above():
    df = build_candles(
        [
            (105.0, 106.0, 104.0, 104.0),
            (104.0, 105.0, 103.0, 103.0),
            (103.0, 104.0, 102.0, 102.0),
            (102.0, 102.0, 98.0, 101.0),  # wicks well under 100, closes back above -> bullish sweep
            (101.0, 102.0, 100.0, 101.5),
        ],
        freq="1h",
    )
    event = detect_named_level_sweep(df, "H1", level_price=100.0, direction="bullish", min_wick_pct=10.0)

    assert event is not None
    assert event.direction == "bullish"
    assert event.timestamp == df["timestamp"].iloc[3] + pd.Timedelta(hours=1)


def test_detect_named_level_sweep_returns_none_when_never_swept():
    df = build_candles([(105.0, 106.0, 104.0, 104.0), (104.0, 105.0, 103.0, 103.0)], freq="1h")
    event = detect_named_level_sweep(df, "H1", level_price=50.0, direction="bullish", min_wick_pct=10.0)
    assert event is None


def test_detect_named_level_sweep_rejects_wick_below_min_pct_threshold():
    # low (99.5) is under the level and close (100.3) is back above it, but
    # the wick is only ~12.7% of the candle's full range (0.7 / 5.5) -- well
    # under the 50% threshold
    df = build_candles([(100.2, 105.0, 99.5, 100.3)], freq="1h")
    event = detect_named_level_sweep(df, "H1", level_price=100.0, direction="bullish", min_wick_pct=50.0)
    assert event is None


def test_asia_session_high_low_excludes_the_still_forming_session():
    config = load_config()
    # 96 H1 bars is enough margin to hold >=2 full Asia sessions without the
    # window itself starting mid-session (see mtf_strategies continuation.py/
    # reversal.py's RECENT_H1_LOOKBACK_BARS note -- a too-tight window was
    # found, during the Paso 3 scan, to silently truncate the reference
    # session and return a wrong high/low).
    ts = pd.date_range("2024-01-01 00:00", periods=96, freq="1h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0})

    high, low = compute_asia_session_high_low(df, config.sessions)
    assert high is not None and low is not None  # sanity: some complete Asia session was found


def test_asia_session_high_low_none_with_no_asia_bars_at_all():
    config = load_config()
    ts = pd.date_range("2024-01-01 00:00", periods=3, freq="1h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0})
    high, low = compute_asia_session_high_low(df, config.sessions)
    assert (high, low) == (None, None)
