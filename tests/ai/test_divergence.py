import pandas as pd
import pytest

from tradingai.ai.data.features.divergence import detect_momentum_divergence


def _base_df(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
            "open": [1.1000] * n,
            "high": [1.1000] * n,
            "low": [1.1000] * n,
            "close": [1.1000] * n,
            "volume": [100] * n,
        }
    )


def test_requires_precomputed_columns():
    df = _base_df(10)
    with pytest.raises(ValueError):
        detect_momentum_divergence(df)


def test_detects_bearish_divergence():
    # Swing high 1: precio 1.1050, RSI 70. Swing high 2 (mas tarde): precio 1.1080 (mas
    # alto) pero RSI 55 (mas bajo) -> divergencia bajista (impulso debilitandose).
    n = 10
    df = _base_df(n)
    df["swing_high"] = False
    df["swing_low"] = False
    df["rsi_14"] = 50.0

    df.loc[2, ["swing_high", "high", "rsi_14"]] = [True, 1.1050, 70.0]
    df.loc[7, ["swing_high", "high", "rsi_14"]] = [True, 1.1080, 55.0]

    result = detect_momentum_divergence(df)

    assert result["bearish_momentum_divergence"].iloc[7]
    assert not result["bearish_momentum_divergence"].iloc[2]
    assert not result["bullish_momentum_divergence"].any()


def test_detects_bullish_divergence():
    n = 10
    df = _base_df(n)
    df["swing_high"] = False
    df["swing_low"] = False
    df["rsi_14"] = 50.0

    df.loc[2, ["swing_low", "low", "rsi_14"]] = [True, 1.0950, 25.0]
    df.loc[7, ["swing_low", "low", "rsi_14"]] = [True, 1.0920, 35.0]

    result = detect_momentum_divergence(df)

    assert result["bullish_momentum_divergence"].iloc[7]
    assert not result["bearish_momentum_divergence"].any()
