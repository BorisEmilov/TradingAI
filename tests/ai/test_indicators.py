import numpy as np
import pandas as pd
import pytest

from tradingai.ai.data.features.indicators import add_indicators


def _trending_candles(n: int = 300, step: float = 0.0015) -> pd.DataFrame:
    # Movimiento direccional constante y sin mechas -> ADX debe subir con el tiempo
    # (tendencia fuerte una vez el indicador se "calienta").
    close = 1.1000 + np.arange(n) * step
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
            "open": close,
            "high": close + 0.0002,
            "low": close - 0.0002,
            "close": close,
            "volume": [100] * n,
        }
    )


def _choppy_candles(n: int = 300, amplitude: float = 0.0010) -> pd.DataFrame:
    # Oscila entre dos niveles sin avanzar -> ADX debe quedar bajo (sin tendencia).
    close = 1.1000 + amplitude * np.sin(np.arange(n) * 0.9)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
            "open": close,
            "high": close + 0.0002,
            "low": close - 0.0002,
            "close": close,
            "volume": [100] * n,
        }
    )


def test_strong_trend_regime_flagged_in_sustained_directional_move():
    out = add_indicators(_trending_candles(), include=["atr", "adx"])
    tail = out.iloc[-20:]
    assert tail["regime_strong_trend"].all()
    assert not tail["regime_ranging"].any()


def test_ranging_regime_flagged_in_choppy_oscillation():
    out = add_indicators(_choppy_candles(), include=["atr", "adx"])
    tail = out.iloc[-20:]
    assert tail["regime_ranging"].all()
    assert not tail["regime_strong_trend"].any()


def test_regime_buckets_are_mutually_exclusive():
    out = add_indicators(_trending_candles(), include=["atr", "adx"])
    # Excluye el warm-up de ADX (NaN): la comparacion NaN<20 da False en las 3
    # columnas, sin esto la suma seria 0 en vez de 1 durante esas primeras velas.
    valid = out.dropna(subset=["ADX_14"])
    bucket_sum = (
        valid["regime_ranging"].astype(int)
        + valid["regime_weak_trend"].astype(int)
        + valid["regime_strong_trend"].astype(int)
    )
    assert (bucket_sum == 1).all()


def test_volatility_regime_flags_low_and_high_relative_to_recent_history():
    n = 300
    close = 1.1000 + np.cumsum(np.random.RandomState(0).normal(0, 0.0002, n))
    # Rango normal (100 velas) -> un salto grande de ATR -> vuelve a la normalidad.
    high = close + 0.0003
    low = close - 0.0003
    high[150:160] = close[150:160] + 0.0050  # pico de volatilidad puntual
    low[150:160] = close[150:160] - 0.0050
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
            "open": close, "high": high, "low": low, "close": close, "volume": [100] * n,
        }
    )
    out = add_indicators(df, include=["atr", "adx"])
    assert out["regime_high_volatility"].iloc[150:160].any()


def test_regime_raises_if_atr_pct_missing():
    df = _trending_candles()
    with pytest.raises(ValueError, match="atr_pct"):
        add_indicators(df, include=["adx"])  # sin 'atr' -> falta atr_pct


def test_bias_bullish_when_emas_stacked_upward():
    # Tendencia alcista sostenida -> EMA20 > EMA50 > EMA200 una vez calentadas.
    out = add_indicators(_trending_candles(), include=["ema"])
    tail = out.iloc[-20:]
    assert tail["bias_bullish"].all()
    assert not tail["bias_bearish"].any()


def test_bias_bearish_when_emas_stacked_downward():
    n = 300
    close = 1.1000 - np.arange(n) * 0.0015
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
            "open": close, "high": close + 0.0002, "low": close - 0.0002, "close": close, "volume": [100] * n,
        }
    )
    out = add_indicators(df, include=["ema"])
    tail = out.iloc[-20:]
    assert tail["bias_bearish"].all()
    assert not tail["bias_bullish"].any()


def test_bias_neutral_when_emas_not_cleanly_stacked():
    # Oscilacion sin tendencia sostenida -> en algun tramo las EMAs se entrelazan
    # (cruces de la oscilacion), ni bullish ni bearish limpio todo el tiempo.
    out = add_indicators(_choppy_candles(), include=["ema"])
    valid = out.dropna(subset=["ema_200_pct"])
    assert valid["bias_neutral"].any()


def test_bias_buckets_are_mutually_exclusive_and_exhaustive():
    out = add_indicators(_trending_candles(), include=["ema"])
    valid = out.dropna(subset=["ema_200_pct"])
    bucket_sum = (
        valid["bias_bullish"].astype(int) + valid["bias_bearish"].astype(int) + valid["bias_neutral"].astype(int)
    )
    assert (bucket_sum == 1).all()


def test_bias_does_not_gate_or_remove_any_rows():
    # Es solo una feature mas -- nunca debe filtrar filas ni bloquear nada por si sola.
    df = _trending_candles()
    out = add_indicators(df, include=["ema"])
    assert len(out) == len(df)
