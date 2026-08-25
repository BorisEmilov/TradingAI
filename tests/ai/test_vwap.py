import numpy as np
import pandas as pd

from tradingai.ai.data.features.vwap import add_vwap_features


def test_price_above_vwap_gives_positive_distance():
    n = 20
    ts = pd.date_range("2024-01-01 00:00", periods=n, freq="15min")
    close = np.linspace(1.1000, 1.1050, n)  # tendencia sostenida al alza dentro del dia
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": close + 0.0005,
            "low": close - 0.0005,
            "close": close,
            "volume": [100] * n,
        }
    )

    result = add_vwap_features(df)

    assert result["vwap_dist_pct"].iloc[-1] > 0
    assert result["vwap_zscore"].iloc[-1] > 0
    assert result["above_vwap_upper_1std"].iloc[-1]


def test_vwap_resets_at_new_day():
    n = 10
    day1 = pd.date_range("2024-01-01 22:00", periods=5, freq="15min")
    day2 = pd.date_range("2024-01-02 00:00", periods=5, freq="15min")
    ts = day1.append(day2)
    close = np.array([1.2000] * 5 + [1.1000] * 5)  # salto brusco al cambiar de dia
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": close + 0.0005,
            "low": close - 0.0005,
            "close": close,
            "volume": [100] * n,
        }
    )

    result = add_vwap_features(df)

    # Si el VWAP no reseteara al cambiar de dia, el primer bar del dia 2 (close=1.10)
    # quedaria muy por debajo de un VWAP arrastrado desde 1.20 -> distancia enorme.
    # Con reset correcto, el primer bar del dia 2 es su propio VWAP -> distancia ~0.
    assert abs(result["vwap_dist_pct"].iloc[5]) < 1e-6
