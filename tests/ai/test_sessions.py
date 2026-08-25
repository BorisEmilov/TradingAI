import pandas as pd

from tradingai.ai.data.features.sessions import add_session_features


def _candles_at_hours(hours: list[int]) -> pd.DataFrame:
    n = len(hours)
    timestamps = [pd.Timestamp("2026-08-24") + pd.Timedelta(hours=h) for h in hours]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [1.1] * n,
            "high": [1.1] * n,
            "low": [1.1] * n,
            "close": [1.1] * n,
            "volume": [100] * n,
        }
    )


def test_session_assignment_default_windows():
    # 3am -> Asia, 8am -> Londres (y killzone), 14h -> NY (y killzone, y overlap con Londres), 20h -> NY solo.
    df = _candles_at_hours([3, 8, 14, 20])
    result = add_session_features(df)

    assert result["session_asia"].tolist() == [True, False, False, False]
    assert result["session_london"].tolist() == [False, True, True, False]
    assert result["session_ny"].tolist() == [False, False, True, True]
    assert result["session_overlap_london_ny"].tolist() == [False, False, True, False]
    assert result["is_killzone"].tolist() == [False, True, True, False]


def test_custom_windows_override_defaults():
    df = _candles_at_hours([5])
    result = add_session_features(df, config={"asia_utc": [0, 6]})
    assert result["session_asia"].iloc[0]


def _candles(rows: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    """rows: (timestamp ISO, high, low, close)."""
    return pd.DataFrame(
        {
            "timestamp": [pd.Timestamp(r[0]) for r in rows],
            "open": [r[3] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [100] * len(rows),
        }
    )


def test_asian_range_builds_progressively_and_freezes_after_session():
    df = _candles(
        [
            ("2026-08-24T00:00", 1.1010, 1.0990, 1.1000),  # Asia: rango parcial 1.1010/1.0990
            ("2026-08-24T01:00", 1.1020, 1.0980, 1.1000),  # Asia: rango se amplia a 1.1020/1.0980
            ("2026-08-24T06:00", 1.1015, 1.0985, 1.1000),  # Asia: dentro del rango ya visto, no cambia
            ("2026-08-24T08:00", 1.1030, 1.1025, 1.1030),  # Londres: rango congelado en 1.1020/1.0980
        ]
    )
    result = add_session_features(df)

    assert result["asian_range_high"].tolist() == [1.1010, 1.1020, 1.1020, 1.1020]
    assert result["asian_range_low"].tolist() == [1.0990, 1.0980, 1.0980, 1.0980]


def test_broke_asian_range_high_and_low_detected_outside_session():
    df = _candles(
        [
            ("2026-08-24T00:00", 1.1010, 1.0990, 1.1000),
            ("2026-08-24T01:00", 1.1020, 1.0980, 1.1000),
            ("2026-08-24T08:00", 1.1030, 1.1025, 1.1030),  # cierre 1.1030 > rango asiatico (1.1020)
            ("2026-08-24T09:00", 1.0975, 1.0970, 1.0975),  # cierre 1.0975 < rango asiatico (1.0980)
        ]
    )
    result = add_session_features(df)

    assert result["broke_asian_range_high"].tolist() == [False, False, True, False]
    assert result["broke_asian_range_low"].tolist() == [False, False, False, True]


def test_asian_range_does_not_leak_across_days():
    df = _candles(
        [
            ("2026-08-24T01:00", 1.1020, 1.0980, 1.1000),  # dia 1: rango 1.1020/1.0980
            ("2026-08-25T00:00", 1.0900, 1.0890, 1.0895),  # dia 2: rango nuevo, no hereda el de ayer
        ]
    )
    result = add_session_features(df)

    assert result["asian_range_high"].tolist() == [1.1020, 1.0900]
    assert result["asian_range_low"].tolist() == [1.0980, 1.0890]


def test_ny_opening_range_breakout_detected():
    df = _candles(
        [
            ("2026-08-24T12:00", 1.1000, 1.0990, 1.0995),  # primera hora de NY (12-13 UTC default)
            ("2026-08-24T14:00", 1.1015, 1.1005, 1.1015),  # fuera del OR -> rompe el alto (1.1000)
        ]
    )
    result = add_session_features(df)

    assert result["broke_ny_or_range_high"].tolist() == [False, True]


def test_range_features_never_produce_nan_even_when_window_never_matches():
    # Simula D1: todas las velas selladas a las 00:00 UTC -> la ventana de apertura de
    # NY (12-13 UTC por defecto) nunca se cumple en ningun dia. Sin fallback, la
    # columna quedaria 100% NaN y normalize_ohlcv descartaria el dataframe entero
    # (bug real visto el 2026-08-25: D1 se quedaba con 0 velas).
    df = _candles(
        [
            ("2026-08-20T00:00", 1.1000, 1.0950, 1.0980),
            ("2026-08-21T00:00", 1.1050, 1.1000, 1.1020),
            ("2026-08-22T00:00", 1.1100, 1.1050, 1.1080),
        ]
    )
    result = add_session_features(df)

    assert not result["ny_or_range_high"].isna().any()
    assert not result["ny_or_range_low"].isna().any()
    assert not result["broke_ny_or_range_high"].isna().any()


def test_killzone_liquidity_sweep_combines_sweep_and_killzone_columns():
    from tradingai.ai.data.features.pipeline import build_feature_pipeline

    n = 120
    timestamps = [pd.Timestamp("2026-08-24T00:00") + pd.Timedelta(minutes=15 * i) for i in range(n)]
    close = [1.1000 + 0.0001 * i for i in range(n)]
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": [c + 0.0002 for c in close],
            "low": [c - 0.0002 for c in close],
            "close": close,
            "volume": [100] * n,
        }
    )
    # Vela en killzone de Londres (8h UTC) con una mecha que barre el maximo reciente
    # y cierra de vuelta dentro del rango -> liquidity_sweep_bearish esperado ahi.
    idx = 32  # timestamp = 08:00 UTC
    df.loc[idx, "high"] = df["high"].iloc[max(0, idx - 100):idx].max() * 1.001
    df.loc[idx, "close"] = close[idx]

    result = build_feature_pipeline(df, config={"features": {}})

    assert "killzone_liquidity_sweep_bearish" in result.columns
    assert result.loc[idx, "killzone_liquidity_sweep_bearish"] == (
        result.loc[idx, "liquidity_sweep_bearish"] and result.loc[idx, "is_killzone"]
    )
