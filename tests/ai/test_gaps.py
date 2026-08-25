import pandas as pd

from tradingai.ai.data.features.gaps import detect_inverted_fvg


def test_detects_bullish_fvg_inversion_causally():
    # Bloque plano (0-3) -> salto alcista sostenido (4-11, deja un FVG sin llenar) ->
    # crash (12) que atraviesa el hueco de vuelta -> debe invalidar el FVG justo en esa vela,
    # no antes (causal: usa solo datos hasta la vela actual).
    open_ = [1.1000] * 4 + [1.1010] * 8 + [1.0950] * 4
    close = list(open_)
    high = [1.1002] * 4 + [1.1012] * 8 + [1.0952] * 4
    low = [1.0998] * 4 + [1.1008] * 8 + [0.9999] * 4

    n = len(open_)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": [100] * n,
        }
    )

    result = detect_inverted_fvg(df, min_gap_size_pips=2.0, pip_size=0.0001)

    assert result["bullish_fvg_inverted"].iloc[12]
    assert not result["bullish_fvg_inverted"].iloc[:12].any()


def test_no_inversion_when_price_never_returns():
    # Salto alcista que se mantiene (nunca vuelve a cerrar por debajo del hueco) -> sin inversion.
    n = 12
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
            "open": [1.1000] * 4 + [1.1010] * 8,
            "high": [1.1002] * 4 + [1.1012] * 8,
            "low": [1.0998] * 4 + [1.1008] * 8,
            "close": [1.1000] * 4 + [1.1010] * 8,
            "volume": [100] * n,
        }
    )

    result = detect_inverted_fvg(df, min_gap_size_pips=2.0, pip_size=0.0001)

    assert not result["bullish_fvg_inverted"].any()
    assert not result["bearish_fvg_inverted"].any()
