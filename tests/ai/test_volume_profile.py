import numpy as np
import pandas as pd

from tradingai.ai.data.features.volume_profile import rolling_volume_profile_features


def test_high_volume_cluster_becomes_poc():
    rng = np.random.default_rng(0)
    n = 120
    window = 100
    close = 1.1000 + rng.normal(0, 0.0005, n)
    high = close + 0.0003
    low = close - 0.0003
    volume = np.full(n, 10.0)

    # Concentra mucho mas volumen alrededor de 1.1000 que en el resto del rango.
    close[:] = np.where(np.arange(n) % 3 == 0, 1.1000, close)
    volume[np.arange(n) % 3 == 0] = 500.0

    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
                        "open": close, "high": high, "low": low, "close": close, "volume": volume})

    result = rolling_volume_profile_features(df, window=window, bins=20)
    last = result.iloc[-1]

    assert abs(last["dist_to_poc_pct"]) < 0.005  # el precio final (1.1000ish) cerca del POC dominante
    assert result["dist_to_poc_pct"].iloc[: window - 1].eq(0.0).all()  # sin historia suficiente -> neutro


def test_inside_value_area_flag_is_boolean():
    # Ventana de referencia (excluye la vela actual) con ruido pequeno centrado en
    # 1.1000; la vela actual tambien cae justo en ese centro -> debe quedar dentro de
    # la value area calculada sobre las velas anteriores.
    rng = np.random.default_rng(1)
    n = 150
    close = 1.1000 + rng.normal(0, 0.0002, n)
    close[-1] = 1.1000
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
            "open": close,
            "high": close + 0.0005,
            "low": close - 0.0005,
            "close": close,
            "volume": [100.0] * n,
        }
    )
    result = rolling_volume_profile_features(df, window=100, bins=10)
    assert result["inside_value_area"].dtype == bool
    assert result["inside_value_area"].iloc[-1]
