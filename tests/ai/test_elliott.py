import numpy as np
import pandas as pd
import pytest

from tradingai.ai.data.features.elliott import compute_elliott_features


def _impulse_df(pivot_prices: list[float], n_per_leg: int = 20) -> pd.DataFrame:
    prices = []
    for a, b in zip(pivot_prices[:-1], pivot_prices[1:]):
        prices.extend(np.linspace(a, b, n_per_leg, endpoint=False))
    prices.extend([pivot_prices[-1]] * 5)
    prices = np.array(prices)
    n = len(prices)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
            "open": prices,
            "close": prices,
            "high": prices + 0.05,
            "low": prices - 0.05,
            "volume": [100] * n,
        }
    )


def test_valid_impulse_scores_full_confidence():
    # P0=100 -> P1=110 (onda1) -> P2=104 (onda2, retrocede 60%) -> P3=130 (onda3, 2.6x
    # onda1) -> P4=120 (onda4, retrocede 38.5% de onda3, sin solapar onda1) -> P5=140.
    df = _impulse_df([100, 110, 104, 130, 120, 140, 130])
    result = compute_elliott_features(df, deviation_pct=0.5)
    row = result.iloc[-1]

    assert row["elliott_wave2_retrace_pct"] == pytest.approx(0.6, abs=0.02)
    assert row["elliott_wave3_extension_ratio"] == pytest.approx(2.6, abs=0.02)
    assert row["elliott_wave4_retrace_pct"] == pytest.approx(0.385, abs=0.02)
    assert not row["elliott_wave4_overlap"]
    assert row["elliott_direction"] == 1
    assert row["elliott_impulse_confidence"] == 1.0


def test_wave4_overlap_penalizes_confidence():
    # Onda4 termina en 108, DENTRO del territorio de onda1 (100-110) -> invalida el conteo.
    df = _impulse_df([100, 110, 104, 130, 108, 140, 130])
    result = compute_elliott_features(df, deviation_pct=0.5)
    row = result.iloc[-1]

    assert row["elliott_wave4_overlap"]
    assert row["elliott_impulse_confidence"] < 1.0


def test_no_spurious_pivot_from_bootstrap_noise():
    # Regresion: al arrancar, max y min parten del mismo precio semilla: un ruido
    # bidireccional minusculo en la primera vela no debe confirmar un pivote espurio
    # en la semilla (bug encontrado: sin el fix, esto rompia la ventana de 6 pivotes
    # y desalineaba todas las ratios subsiguientes).
    df = _impulse_df([100, 110, 104, 130, 120, 140, 130])
    result = compute_elliott_features(df, deviation_pct=0.5)
    # Con el fix, el conteo final debe coincidir exactamente con el impulso limpio
    # (ver test_valid_impulse_scores_full_confidence) y no con valores degenerados.
    assert result["elliott_impulse_confidence"].max() == 1.0
