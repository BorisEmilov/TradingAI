import pandas as pd
import pytest

from trader.events import TF_DURATION, closed_candles_as_of, ensure_utc_sorted, events_up_to, MarketEvent
from tests.conftest import build_candles


def test_ensure_utc_sorted_rejects_naive_timestamps():
    df = build_candles([(1, 1, 1, 1)])
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        ensure_utc_sorted(df)


def test_ensure_utc_sorted_rejects_unsorted():
    df = build_candles([(1, 1, 1, 1), (1, 1, 1, 1)])
    df = df.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="sorted ascending"):
        ensure_utc_sorted(df)


def test_closed_candles_as_of_excludes_forming_bar():
    df = build_candles([(1, 1, 1, 1)] * 4, freq="15min")
    # bar 3 opens at +45min and closes at +60min on M15
    as_of = df["timestamp"].iloc[3] + pd.Timedelta(minutes=1)
    closed = closed_candles_as_of(df, "M15", as_of)
    assert len(closed) == 3

    as_of_after_close = df["timestamp"].iloc[3] + TF_DURATION["M15"]
    closed_full = closed_candles_as_of(df, "M15", as_of_after_close)
    assert len(closed_full) == 4


def test_events_up_to_filters_by_own_timestamp():
    ts = pd.Timestamp("2026-01-01", tz="UTC")
    e1 = MarketEvent.point("bos", "H1", ts, "bullish", 1.0)
    e2 = MarketEvent.point("bos", "H1", ts + pd.Timedelta(hours=1), "bullish", 1.0)
    result = events_up_to([e1, e2], ts)
    assert result == [e1]
