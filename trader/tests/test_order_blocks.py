import pandas as pd

from trader.detectors.order_blocks import detect_order_blocks
from trader.events import TF_DURATION, MarketEvent
from tests.conftest import build_candles

_H1 = TF_DURATION["H1"]


def test_bullish_ob_scans_back_past_bullish_pause_bars_to_last_bearish_candle():
    bars = [
        (10.0, 10.1, 9.7, 9.8),  # filler, bearish
        (9.8, 10.0, 9.6, 9.7),  # the real OB candle (bearish)
        (9.7, 9.9, 9.65, 9.85),  # small bullish pause -- must be skipped over
        (9.85, 12.0, 9.8, 11.9),  # displacement candle, triggers BOS
    ]
    df = build_candles(bars, freq="1h")
    atr = pd.Series([1.0, 1.0, 1.0, 1.0])
    structure_events = [
        MarketEvent.point("bos", "H1", df["timestamp"].iloc[3] + _H1, "bullish", 11.9),
    ]

    obs = detect_order_blocks(df, "H1", structure_events, atr, displacement_atr_multiple=1.5)

    assert len(obs) == 1
    ob = obs[0]
    assert ob.kind == "order_block_bullish"
    assert ob.price_high == 10.0
    assert ob.price_low == 9.6
    assert ob.timestamp == df["timestamp"].iloc[3] + _H1


def test_bearish_ob_symmetric_case():
    bars = [
        (9.7, 10.1, 9.6, 10.0),  # bullish filler
        (10.0, 10.4, 9.9, 10.3),  # the real OB candle (bullish)
        (10.3, 10.35, 8.0, 8.1),  # displacement candle down, triggers BOS
    ]
    df = build_candles(bars, freq="1h")
    atr = pd.Series([1.0, 1.0, 1.0])
    structure_events = [
        MarketEvent.point("bos", "H1", df["timestamp"].iloc[2] + _H1, "bearish", 8.1),
    ]

    obs = detect_order_blocks(df, "H1", structure_events, atr, displacement_atr_multiple=1.5)

    assert len(obs) == 1
    ob = obs[0]
    assert ob.kind == "order_block_bearish"
    assert ob.price_high == 10.4
    assert ob.price_low == 9.9


def test_weak_break_below_displacement_threshold_yields_no_ob():
    bars = [
        (9.8, 10.0, 9.6, 9.7),
        (9.7, 9.9, 9.6, 9.8),  # small range break, not a real impulse
    ]
    df = build_candles(bars, freq="1h")
    atr = pd.Series([1.0, 1.0])
    structure_events = [
        MarketEvent.point("bos", "H1", df["timestamp"].iloc[1] + _H1, "bullish", 9.8),
    ]

    obs = detect_order_blocks(df, "H1", structure_events, atr, displacement_atr_multiple=1.5)
    assert obs == []
