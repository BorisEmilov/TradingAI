import pandas as pd

from tradingai.ai.data.features.smc import detect_liquidity_sweeps


def _flat_ohlc(n: int, high: float, low: float, close: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
            "open": [close] * n,
            "high": [high] * n,
            "low": [low] * n,
            "close": [close] * n,
            "volume": [100] * n,
        }
    )


def test_detects_bearish_liquidity_sweep():
    df = _flat_ohlc(12, high=1.1005, low=1.0995, close=1.1000)
    # Vela 10: mecha perfora el maximo reciente (1.1005) pero cierra de vuelta abajo -> rechazo bajista.
    df.loc[10, "high"] = 1.1050
    df.loc[10, "close"] = 1.0998

    result = detect_liquidity_sweeps(df, lookback=5, tolerance_pct=0.0)

    assert result["liquidity_sweep_bearish"].iloc[10]
    assert not result["liquidity_sweep_bearish"].iloc[:10].any()
    assert not result["liquidity_sweep_bullish"].any()


def test_detects_bullish_liquidity_sweep():
    df = _flat_ohlc(12, high=1.1005, low=1.0995, close=1.1000)
    # Vela 10: mecha perfora el minimo reciente (1.0995) pero cierra de vuelta arriba -> rechazo alcista.
    df.loc[10, "low"] = 1.0950
    df.loc[10, "close"] = 1.1002

    result = detect_liquidity_sweeps(df, lookback=5, tolerance_pct=0.0)

    assert result["liquidity_sweep_bullish"].iloc[10]
    assert not result["liquidity_sweep_bullish"].iloc[:10].any()
    assert not result["liquidity_sweep_bearish"].any()


def test_no_sweep_without_reversal_close():
    # Perfora el maximo pero CIERRA por encima (continuacion, no rechazo) -> no es sweep.
    df = _flat_ohlc(12, high=1.1005, low=1.0995, close=1.1000)
    df.loc[10, "high"] = 1.1050
    df.loc[10, "close"] = 1.1040

    result = detect_liquidity_sweeps(df, lookback=5, tolerance_pct=0.0)

    assert not result["liquidity_sweep_bearish"].any()
    assert not result["liquidity_sweep_bullish"].any()
