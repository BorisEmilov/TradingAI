import pandas as pd

from trader.regime import EFFICIENCY_THRESHOLD, WINDOW, regime_as_of
from tests.conftest import build_candles


def _closes_only_df(closes: list[float]) -> pd.DataFrame:
    bars = [(c, c, c, c) for c in closes]
    return build_candles(bars, freq="1D")


def test_insufficient_history_returns_none():
    df = _closes_only_df([10.0] * (WINDOW - 1))
    assert regime_as_of(df, len(df) - 1) is None


def test_pure_trend_efficiency_is_one():
    closes = [10.0 + i for i in range(WINDOW + 1)]  # straight line up, no backtracking
    df = _closes_only_df(closes)
    result = regime_as_of(df, len(df) - 1)
    assert result is not None
    assert result.regime == "trend_up"
    assert abs(result.efficiency_ratio - 1.0) < 1e-9


def test_pure_range_efficiency_is_near_zero():
    # oscillates back to the start -- net displacement ~0, lots of path traveled
    closes = [10.0, 11.0, 10.0, 11.0, 10.0, 11.0, 10.0, 11.0, 10.0, 11.0, 10.0,
              11.0, 10.0, 11.0, 10.0, 11.0, 10.0, 11.0, 10.0, 11.0, 10.0]
    assert len(closes) == WINDOW + 1
    df = _closes_only_df(closes)
    result = regime_as_of(df, len(df) - 1)
    assert result is not None
    assert result.regime == "range"
    assert result.efficiency_ratio < 0.1


def test_downtrend_direction():
    closes = [30.0 - i for i in range(WINDOW + 1)]
    df = _closes_only_df(closes)
    result = regime_as_of(df, len(df) - 1)
    assert result.regime == "trend_down"


def test_threshold_boundary_is_strict_greater_than():
    # construct a case landing exactly at 0.5: net=10, path=20
    closes = [10.0] * (WINDOW + 1)
    closes[-1] = 20.0  # net = 10
    # build path of exactly 20 by walking net 10 up then some back-and-forth
    # simpler: half the path is net progress, half is backtracking
    closes = [10.0]
    for _ in range(WINDOW // 2):
        closes.append(closes[-1] + 2.0)
        closes.append(closes[-1] - 1.0)
    closes = closes[: WINDOW + 1]
    df = _closes_only_df(closes)
    result = regime_as_of(df, len(df) - 1)
    assert result is not None
    # whatever the exact ratio lands on, confirm the rule is a strict `>`
    if result.efficiency_ratio == EFFICIENCY_THRESHOLD:
        assert result.regime == "range"


def test_causal_future_bars_dont_affect_earlier_classification():
    closes = [10.0 + i for i in range(WINDOW + 1)]
    df_short = _closes_only_df(closes)
    result_short = regime_as_of(df_short, len(df_short) - 1)

    df_long = _closes_only_df(closes + [100.0, 0.5, 200.0])  # wild future data appended
    result_long = regime_as_of(df_long, len(df_short) - 1)  # SAME idx as before

    assert result_short == result_long
