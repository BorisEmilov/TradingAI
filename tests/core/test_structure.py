import pandas as pd

from tradingai.core.structure import last_confirmed_swing_high, last_confirmed_swing_low


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
