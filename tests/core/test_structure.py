import pandas as pd
import pytest

from tradingai.core.structure import (
    confirmed_swing_highs,
    confirmed_swing_lows,
    last_confirmed_swing_high,
    last_confirmed_swing_low,
)


def _candles(lows: list[float], highs: list[float] | None = None) -> pd.DataFrame:
    highs = highs if highs is not None else [low + 0.0010 for low in lows]
    return pd.DataFrame({"low": lows, "high": highs})


def test_finds_confirmed_swing_low():
    lows = [1.1030, 1.1020, 1.1010, 1.0995, 1.1005, 1.1015, 1.1025, 1.1035]
    assert last_confirmed_swing_low(_candles(lows)) == 1.0995


def test_finds_confirmed_swing_high():
    highs = [1.0970, 1.0980, 1.0990, 1.1005, 1.0995, 1.0985, 1.0975, 1.0965]
    candles = _candles(lows=[h - 0.0010 for h in highs], highs=highs)
    assert last_confirmed_swing_high(candles) == 1.1005


def test_returns_none_when_no_swing_confirmed():
    lows = [1.1000, 1.0990, 1.0980, 1.0970, 1.0960, 1.0950, 1.0940]
    assert last_confirmed_swing_low(_candles(lows)) is None


def test_confirmed_swing_lows_returns_multiple_in_chronological_order():
    # Dos swing lows confirmados: 1.0995 (mas viejo) y 1.0980 (mas reciente, mas bajo
    # -- una ruptura de estructura si se esta en largo).
    lows = [1.1030, 1.1020, 1.1010, 1.0995, 1.1005, 1.1015, 1.1025, 1.1035, 1.1045, 1.1030, 1.0980, 1.0990, 1.1000, 1.1010]
    found = confirmed_swing_lows(_candles(lows), count=2)
    assert found == [1.0995, 1.0980]


def test_confirmed_swing_highs_returns_multiple_in_chronological_order():
    highs = [h * -1 + 2.2 for h in [1.1030, 1.1020, 1.1010, 1.0995, 1.1005, 1.1015, 1.1025, 1.1035, 1.1045, 1.1030, 1.0980, 1.0990, 1.1000, 1.1010]]
    lows = [h - 0.0010 for h in highs]
    found = confirmed_swing_highs(pd.DataFrame({"low": lows, "high": highs}), count=2)
    assert found == pytest.approx([1.1005, 1.1020])
