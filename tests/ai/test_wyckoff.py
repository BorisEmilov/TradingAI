import pandas as pd

from tradingai.ai.data.features.wyckoff import detect_wyckoff_events


def _range_bound_df(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
            "open": [1.1000] * n,
            "high": [1.1005] * n,
            "low": [1.0995] * n,
            "close": [1.1000] * n,
            "volume": [100] * n,
        }
    )


def test_detects_spring():
    df = _range_bound_df(25)
    df.loc[22, "low"] = 1.0950  # barre por debajo del soporte del rango
    df.loc[22, "close"] = 1.0998  # pero cierra de vuelta dentro

    result = detect_wyckoff_events(df, range_window=20, tolerance_pct=0.0)

    assert result["wyckoff_spring"].iloc[22]
    assert result["wyckoff_spring"].sum() == 1
    assert not result["wyckoff_upthrust"].any()


def test_detects_upthrust():
    df = _range_bound_df(25)
    df.loc[22, "high"] = 1.1050  # rompe la resistencia del rango
    df.loc[22, "close"] = 1.1002  # pero cierra de vuelta dentro

    result = detect_wyckoff_events(df, range_window=20, tolerance_pct=0.0)

    assert result["wyckoff_upthrust"].iloc[22]
    assert not result["wyckoff_spring"].any()


def test_range_uses_prior_bars_only_not_current():
    # El rango de referencia excluye la vela actual (.shift(1)): una vela que ella misma
    # marca un nuevo extremo no puede "romper su propio" rango en la misma vela.
    df = _range_bound_df(25)
    df.loc[22, "low"] = 1.0950
    df.loc[22, "close"] = 1.0950  # cierra en el propio extremo, no hay reclaim -> no es spring

    result = detect_wyckoff_events(df, range_window=20, tolerance_pct=0.0)

    assert not result["wyckoff_spring"].iloc[22]
