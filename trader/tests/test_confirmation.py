import pandas as pd

from tests.conftest import build_candles
from trader.confirmation import CONFIRMATION_THRESHOLD, ROLLING_WINDOW, _wick_and_close_position, compute_confirmations


def test_wick_and_close_position_bullish_pin_bar():
    # long lower wick, close near the high -> strong bullish rejection
    wick_ratio, close_position = _wick_and_close_position(o=9.5, h=10.0, l=8.0, c=9.8, direction="bullish")
    assert wick_ratio > 0.6
    assert close_position > 0.8


def test_wick_and_close_position_bearish_pin_bar():
    wick_ratio, close_position = _wick_and_close_position(o=9.5, h=11.0, l=9.0, c=9.3, direction="bearish")
    assert wick_ratio > 0.5
    assert close_position > 0.7


def test_wick_and_close_position_zero_range_is_safe():
    wick_ratio, close_position = _wick_and_close_position(o=10.0, h=10.0, l=10.0, c=10.0, direction="bullish")
    assert wick_ratio == 0.0
    assert close_position == 0.5


def test_below_min_window_never_confirms():
    # Fewer candles than ROLLING_WINDOW -> no percentile ever computed -> no confirmation possible.
    bars = [(10, 10.5, 9.5, 10) for _ in range(ROLLING_WINDOW - 1)]
    df = build_candles(bars, freq="15min")
    results = compute_confirmations(df)
    assert all(v["bullish"] is None and v["bearish"] is None for v in results.values())


def test_strong_rejection_with_followthrough_confirms():
    # Fill history with weak/neutral candles (small wicks, close mid-range),
    # then plant one clean bullish pin bar followed by an up-close continuation.
    filler = [(10.0, 10.1, 9.9, 10.0) for _ in range(ROLLING_WINDOW)]
    pin_bar = (9.6, 10.0, 9.0, 9.95)  # long lower wick, close near high -> top of the filler distribution
    followthrough = (9.95, 10.5, 9.9, 10.4)  # closes higher than the pin bar's close
    df = build_candles(filler + [pin_bar, followthrough], freq="15min")
    results = compute_confirmations(df)
    pin_idx = len(filler)
    bullish_result = results[pin_idx]["bullish"]
    assert bullish_result is not None
    assert bullish_result.followthrough is True
    assert bullish_result.score_percentile >= CONFIRMATION_THRESHOLD


def test_strong_rejection_without_followthrough_does_not_confirm():
    filler = [(10.0, 10.1, 9.9, 10.0) for _ in range(ROLLING_WINDOW)]
    pin_bar = (9.6, 10.0, 9.0, 9.95)
    no_followthrough = (9.95, 10.0, 9.5, 9.6)  # closes LOWER than the pin bar's close -> no bullish followthrough
    df = build_candles(filler + [pin_bar, no_followthrough], freq="15min")
    results = compute_confirmations(df)
    pin_idx = len(filler)
    assert results[pin_idx]["bullish"] is None


def test_last_candle_has_no_followthrough_data_and_cannot_confirm():
    filler = [(10.0, 10.1, 9.9, 10.0) for _ in range(ROLLING_WINDOW)]
    pin_bar = (9.6, 10.0, 9.0, 9.95)
    df = build_candles(filler + [pin_bar], freq="15min")
    results = compute_confirmations(df)
    last_idx = len(df) - 1
    assert results[last_idx]["bullish"] is None


def test_candle_below_absolute_floor_never_confirms():
    filler = [(10.0, 10.1, 9.9, 10.0) for _ in range(ROLLING_WINDOW)]
    tiny_pin_bar = (9.999, 10.0, 9.998, 9.9995)  # clean shape, but 0.002 range -- below the 0.05 floor
    followthrough = (9.9995, 10.5, 9.9, 10.4)
    df = build_candles(filler + [tiny_pin_bar, followthrough], freq="15min")
    results = compute_confirmations(df, min_candle_range=0.05)
    pin_idx = len(filler)
    assert results[pin_idx]["bullish"] is None
    assert results[pin_idx]["bearish"] is None


def test_below_floor_candles_are_excluded_from_the_rolling_sample():
    # If a below-floor candle were still inserted into the tracker, it would
    # count toward the ROLLING_WINDOW sample without ever being a valid
    # confirmation candidate itself -- diluting/skewing the comparison
    # population. Fill with ROLLING_WINDOW tiny (below-floor) candles first;
    # since none of them count, a genuinely large candle right after should
    # still be BELOW the min-sample requirement (no percentile defined yet),
    # not sitting on a full window of noise.
    tiny_filler = [(10.0, 10.001, 9.999, 10.0) for _ in range(ROLLING_WINDOW)]  # range=0.002, below floor
    pin_bar = (9.6, 10.0, 9.0, 9.95)
    followthrough = (9.95, 10.5, 9.9, 10.4)
    df = build_candles(tiny_filler + [pin_bar, followthrough], freq="15min")
    results = compute_confirmations(df, min_candle_range=0.05)
    pin_idx = len(tiny_filler)
    # Only 0 qualifying candles preceded it (all filler was below floor) --
    # nowhere near the 500 required for a defined percentile.
    assert results[pin_idx]["bullish"] is None


def test_zero_floor_preserves_old_behavior():
    filler = [(10.0, 10.1, 9.9, 10.0) for _ in range(ROLLING_WINDOW)]
    pin_bar = (9.6, 10.0, 9.0, 9.95)
    followthrough = (9.95, 10.5, 9.9, 10.4)
    df = build_candles(filler + [pin_bar, followthrough], freq="15min")
    with_default = compute_confirmations(df)
    with_explicit_zero = compute_confirmations(df, min_candle_range=0.0)
    assert with_default == with_explicit_zero


def test_causal_future_bars_dont_affect_earlier_confirmation():
    filler = [(10.0, 10.1, 9.9, 10.0) for _ in range(ROLLING_WINDOW)]
    pin_bar = (9.6, 10.0, 9.0, 9.95)
    followthrough = (9.95, 10.5, 9.9, 10.4)
    bars_short = filler + [pin_bar, followthrough]
    df_short = build_candles(bars_short, freq="15min")
    results_short = compute_confirmations(df_short)

    extra_future = [(20.0, 25.0, 1.0, 3.0), (3.0, 4.0, 0.5, 0.6)]  # wild, unrelated future candles
    df_long = build_candles(bars_short + extra_future, freq="15min")
    results_long = compute_confirmations(df_long)

    pin_idx = len(filler)
    assert results_short[pin_idx]["bullish"] == results_long[pin_idx]["bullish"]
