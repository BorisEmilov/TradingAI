import pandas as pd

from trader.backtest.engine import _candidate_confirmation_timestamps
from trader.events import TF_DURATION, MarketEvent
from trader.pipeline.engine import MultiTimeframePipeline, TimeframeAnalysis
from tests.conftest import build_candles

_FLAT_BAR = (9.8, 9.85, 9.75, 9.8)


def _m15_analysis_with_one_confirmation(n_bars: int, event_bar_idx: int) -> tuple[TimeframeAnalysis, pd.Timestamp]:
    df = build_candles([_FLAT_BAR] * n_bars, freq="15min")
    event_ts = df["timestamp"].iloc[event_bar_idx] + TF_DURATION["M15"]
    event = MarketEvent.point("choch", "M15", event_ts, "bullish", 9.8)
    analysis = TimeframeAnalysis(
        timeframe="M15", df=df, swings=[], structure_events=[event], order_blocks=[], inverted_order_blocks=[],
        fvgs=[], inverted_fvgs=[], sweeps=[], equal_levels=[], turtle_soups=[], sharp_turns=[],
        support_resistance=[], elliott=[],
    )
    return analysis, event_ts


def test_candidate_timestamps_window_1_is_exactly_the_event_bar():
    analysis, event_ts = _m15_analysis_with_one_confirmation(n_bars=5, event_bar_idx=2)

    candidates = _candidate_confirmation_timestamps(analysis, window_candles=1)

    assert candidates == [event_ts]


def test_candidate_timestamps_window_3_includes_the_two_following_bars():
    analysis, event_ts = _m15_analysis_with_one_confirmation(n_bars=5, event_bar_idx=2)

    candidates = _candidate_confirmation_timestamps(analysis, window_candles=3)

    assert candidates == [event_ts, event_ts + TF_DURATION["M15"], event_ts + 2 * TF_DURATION["M15"]]


def test_latest_confirmation_and_candidate_timestamps_are_synchronized_one_candle_stale():
    """Regression guard for the exact bug prompt-implementar-m15-n2.md called
    out: `_latest_confirmation` (decides whether a signal fires) and
    `_candidate_confirmation_timestamps` (decides which bars the backtest
    even visits) each apply their own window -- if those two windows ever
    drift apart, the backtest would silently skip bars that live production
    logic considers valid. This builds a confirmation that's 1 candle stale
    by the "as of" bar and checks that BOTH mechanisms agree at every
    window_candles value: window=1 says no (both), window=2 says yes (both),
    and critically, the exact bar `_latest_confirmation` would accept under
    window=2 is present in `_candidate_confirmation_timestamps`'s output for
    that same window -- otherwise the backtest loop would never reach it.
    """
    full_analysis, event_ts = _m15_analysis_with_one_confirmation(n_bars=5, event_bar_idx=2)
    stale_by_one_ts = event_ts + TF_DURATION["M15"]  # the bar 1 candle after the confirmation

    # Truncate the analysis to represent "as of stale_by_one_ts" -- df ends
    # there, but the confirmation event (at event_ts <= stale_by_one_ts)
    # stays visible, exactly like a real causal as_of() truncation would.
    truncated_df = full_analysis.df[full_analysis.df["timestamp"] + TF_DURATION["M15"] <= stale_by_one_ts]
    truncated_analysis = TimeframeAnalysis(
        timeframe="M15", df=truncated_df, swings=[], structure_events=full_analysis.structure_events,
        order_blocks=[], inverted_order_blocks=[], fvgs=[], inverted_fvgs=[], sweeps=[], equal_levels=[],
        turtle_soups=[], sharp_turns=[], support_resistance=[], elliott=[],
    )

    for window_candles, should_find in ((1, False), (2, True)):
        found = MultiTimeframePipeline._latest_confirmation(truncated_analysis, "bullish", window_candles)
        assert (found is not None) == should_find, f"window_candles={window_candles}"

        visited = _candidate_confirmation_timestamps(full_analysis, window_candles)
        # The bar _latest_confirmation would need to be evaluated AT (stale_by_one_ts)
        # must be among the bars the backtest loop actually visits under the SAME
        # window -- that's the synchronization the two independent mechanisms need.
        assert (stale_by_one_ts in visited) == should_find, f"window_candles={window_candles}"
