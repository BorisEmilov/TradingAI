"""Regresión de un bug real encontrado en la auditoría visual del Paso 3
(2026-09-20): `reversal.py` buscaba el MSS en la dirección OPUESTA a
`direction4h` ("bullish" para un SHORT). La regla 3 de Estrategia 2 es
explícita para el caso SHORT: "MSS bajista" y "FVG bajista" tras un sweep de
liquidez superior -- MISMA dirección que el trade, no la opuesta. El síntoma
que lo delató: el 100% de las señales de reversión auditadas mostraban una
FVG "no genuina" (ver scripts/audit_mtf_signals.py). `continuation.py` ya
pasaba `direction4h` directo a `first_post_sweep_mss` (correcto); esto era
específico de `reversal.py`.

En vez de armar un escenario sintético que complete TODO el gate (extremo 4H
+ extensión + sweep + MSS + FVG + retroceso + piso de riesgo + R:R, varios
intentos anteriores mostraron que es delicado de construir a mano sin
resultar en otro condicional fallando por motivos ajenos al que se quiere
probar) -- se verifica directamente el argumento que `_evaluate_one_setup`
le pasa a `first_post_sweep_mss`, con un mock. Prueba exactamente la línea
que tenía el bug, sin depender de que el resto de la cadena también calce.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from trader.config import load_config
from trader.mtf_strategies.analysis import SimpleAnalysis
from trader.mtf_strategies.reversal import _evaluate_one_setup


def _empty_analysis(timeframe: str, close: float = 100.0) -> SimpleAnalysis:
    df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")], "open": [close], "high": [close],
        "low": [close], "close": [close],
    })
    return SimpleAnalysis(
        timeframe=timeframe, df=df, swings=[], structure_events=[], sweeps=[], equal_levels=[],
        fvgs=[], order_blocks=[], inverted_order_blocks=[], atr=pd.Series([1.0]),
    )


def test_short_setup_requests_bearish_mss_not_bullish():
    """direction4h="bearish" (extremo superior, SHORT) debe pedir MSS
    "bearish" -- el bug pedía "bullish"."""
    config = load_config()
    h4 = _empty_analysis("H4")
    h1 = _empty_analysis("H1")
    m15 = _empty_analysis("M15")
    d1 = h1.df

    with patch("trader.mtf_strategies.reversal.detect_named_level_sweep") as mock_sweep, \
         patch("trader.mtf_strategies.reversal.first_post_sweep_mss") as mock_mss:
        mock_sweep.return_value = type("S", (), {"timestamp": pd.Timestamp("2024-01-01", tz="UTC")})()
        mock_mss.return_value = None  # no importa el resultado, solo el argumento con el que se llama

        _evaluate_one_setup(
            "setup_b_pdh_short", "EURUSD", "bearish", 100.0, "PDH",
            h4, h1, m15, d1, config, pd.Timestamp("2024-01-01", tz="UTC"), "london", 1.0,
            asia_high=None, asia_low=None,
        )

        assert mock_mss.called
        called_direction = mock_mss.call_args.args[3]
        assert called_direction == "bearish", f"esperaba MSS bearish para un SHORT, pidió {called_direction!r}"


def test_long_setup_requests_bullish_mss_not_bearish():
    """direction4h="bullish" (extremo inferior, LONG) debe pedir MSS "bullish"."""
    config = load_config()
    h4 = _empty_analysis("H4")
    h1 = _empty_analysis("H1")
    m15 = _empty_analysis("M15")
    d1 = h1.df

    with patch("trader.mtf_strategies.reversal.detect_named_level_sweep") as mock_sweep, \
         patch("trader.mtf_strategies.reversal.first_post_sweep_mss") as mock_mss:
        mock_sweep.return_value = type("S", (), {"timestamp": pd.Timestamp("2024-01-01", tz="UTC")})()
        mock_mss.return_value = None

        _evaluate_one_setup(
            "setup_b_pdl_long", "EURUSD", "bullish", 100.0, "PDL",
            h4, h1, m15, d1, config, pd.Timestamp("2024-01-01", tz="UTC"), "london", 1.0,
            asia_high=None, asia_low=None,
        )

        called_direction = mock_mss.call_args.args[3]
        assert called_direction == "bullish", f"esperaba MSS bullish para un LONG, pidió {called_direction!r}"
