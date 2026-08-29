import pandas as pd
import pytest

from tradingai.ai.data.features.fibonacci import compute_ote_zone


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
        compute_ote_zone(df)


def test_bullish_impulse_marks_ote_on_retracement():
    # Impulso alcista: swing_low en 1.1000 (idx 2), swing_high en 1.1100 (idx 5).
    # Retroceso del 70% del tramo (dentro de la banda 61.8-78.6%) en idx 8.
    n = 10
    df = _base_df(n)
    df["swing_high"] = False
    df["swing_low"] = False

    df.loc[2, ["swing_low", "low"]] = [True, 1.1000]
    df.loc[5, ["swing_high", "high"]] = [True, 1.1100]
    df.loc[8, "close"] = 1.1100 - 0.70 * (1.1100 - 1.1000)

    result = compute_ote_zone(df)

    assert result["in_ote_bullish"].iloc[8]
    assert not result["in_ote_bearish"].any()
    assert result["fib_retracement_pct"].iloc[8] == pytest.approx(0.70)


def test_bearish_impulse_marks_ote_on_retracement():
    # Impulso bajista: swing_high en 1.1100 (idx 2), swing_low en 1.1000 (idx 5).
    # Retroceso del 65% del tramo en idx 8.
    n = 10
    df = _base_df(n)
    df["swing_high"] = False
    df["swing_low"] = False

    df.loc[2, ["swing_high", "high"]] = [True, 1.1100]
    df.loc[5, ["swing_low", "low"]] = [True, 1.1000]
    df.loc[8, "close"] = 1.1000 + 0.65 * (1.1100 - 1.1000)

    result = compute_ote_zone(df)

    assert result["in_ote_bearish"].iloc[8]
    assert not result["in_ote_bullish"].any()
    assert result["fib_retracement_pct"].iloc[8] == pytest.approx(0.65)


def test_no_zone_without_retracement_into_band():
    # Mismo impulso alcista, pero el precio nunca retrocede hacia la banda dorada
    # (se queda pegado cerca del high) -> no debe marcar nada.
    n = 10
    df = _base_df(n)
    df["swing_high"] = False
    df["swing_low"] = False

    df.loc[2, ["swing_low", "low"]] = [True, 1.1000]
    df.loc[5, ["swing_high", "high"]] = [True, 1.1100]
    df.loc[8, "close"] = 1.1095

    result = compute_ote_zone(df)

    assert not result["in_ote_bullish"].any()
    assert not result["in_ote_bearish"].any()
