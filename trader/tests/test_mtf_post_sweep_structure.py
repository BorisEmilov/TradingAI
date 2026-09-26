from __future__ import annotations

import pandas as pd

from trader.events import MarketEvent
from trader.mtf_strategies.post_sweep_structure import first_post_sweep_mss
from tests.conftest import build_candles


def _swing(kind: str, price: float, ts: pd.Timestamp) -> MarketEvent:
    return MarketEvent.point(kind=kind, timeframe="M15", timestamp=ts, direction="neutral", price=price)


def test_mss_ignores_a_pre_sweep_swing_that_would_have_fired_earlier():
    """A swing high at 102 formed BEFORE the sweep, and a swing high at 108
    formed AFTER it. Price breaks 102 early (right after the sweep) but not
    108 until later. If the pre-sweep swing were (wrongly) considered, MSS
    would fire at the 102 break; it must instead wait for 108."""
    bars = [
        (95.0, 96.0, 94.0, 95.5),  # 0
        (95.5, 97.0, 95.0, 96.5),  # 1  <- pre-sweep swing high pivot forms around here (102 set manually below)
        (96.5, 95.0, 90.0, 91.0),  # 2  sweep candle (wicks low, closes back up) -- SWEEP happens at close of bar 2
        (91.0, 103.0, 90.5, 102.5),  # 3  breaks the OLD pre-sweep level (102) -- must NOT count as MSS
        (102.5, 104.0, 101.0, 103.0),  # 4
        (103.0, 105.0, 102.0, 104.0),  # 5  <- post-sweep swing high pivot (108, set manually) forms around here
        (104.0, 106.0, 103.0, 105.0),  # 6
        (105.0, 109.0, 104.0, 108.5),  # 7  breaks the POST-sweep level (108) -- THIS should be the MSS
        (108.5, 110.0, 108.0, 109.5),  # 8
    ]
    df = build_candles(bars, freq="15min")
    step = pd.Timedelta(minutes=15)
    sweep_ts = df["timestamp"].iloc[2] + step  # close of the sweep candle

    pre_sweep_swing = _swing("swing_high", 102.0, df["timestamp"].iloc[1] + step)  # BEFORE sweep_ts
    post_sweep_swing = _swing("swing_high", 108.0, df["timestamp"].iloc[5] + step)  # AFTER sweep_ts
    assert pre_sweep_swing.timestamp < sweep_ts < post_sweep_swing.timestamp

    mss = first_post_sweep_mss(df, [pre_sweep_swing, post_sweep_swing], sweep_ts, expected_direction="bullish")

    assert mss is not None
    assert mss.price == 108.5  # the close that broke 108, not the one that broke 102
    assert mss.timestamp == df["timestamp"].iloc[7] + step


def test_mss_returns_none_when_no_swing_formed_after_the_sweep():
    bars = [(95.0, 96.0, 94.0, 95.5), (95.5, 110.0, 95.0, 109.0)]
    df = build_candles(bars, freq="15min")
    step = pd.Timedelta(minutes=15)
    sweep_ts = df["timestamp"].iloc[0] + step

    only_pre_sweep_swing = _swing("swing_high", 100.0, df["timestamp"].iloc[0] + step - pd.Timedelta(minutes=1))
    mss = first_post_sweep_mss(df, [only_pre_sweep_swing], sweep_ts, expected_direction="bullish")

    assert mss is None


def test_mss_direction_must_match_expected():
    """A break in the WRONG direction (bearish break right after a bullish-
    expected post-sweep swing) must not be returned as the bullish MSS."""
    bars = [
        (100.0, 101.0, 99.0, 100.0),  # 0
        (100.0, 101.0, 95.0, 96.0),  # 1  <- swing low pivot forms here (for a bearish break test)
        (96.0, 97.0, 94.0, 95.0),  # 2
        (95.0, 96.0, 90.0, 91.0),  # 3  closes below the swing low -> bearish break
    ]
    df = build_candles(bars, freq="15min")
    step = pd.Timedelta(minutes=15)
    sweep_ts = df["timestamp"].iloc[0] + step
    post_sweep_low = _swing("swing_low", 95.0, df["timestamp"].iloc[1] + step)

    mss_bullish = first_post_sweep_mss(df, [post_sweep_low], sweep_ts, expected_direction="bullish")
    assert mss_bullish is None  # the only break available is bearish, not the requested bullish
