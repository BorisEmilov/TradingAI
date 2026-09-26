from __future__ import annotations

import pandas as pd

from trader.mtf_strategies.exits import candles_elapsed_m15, fvg_setup_expired, temporal_exit_triggered

_T0 = pd.Timestamp("2024-01-01 00:00", tz="UTC")


def test_candles_elapsed_m15_exact_count():
    assert candles_elapsed_m15(_T0, _T0 + pd.Timedelta(minutes=15)) == 1
    assert candles_elapsed_m15(_T0, _T0 + pd.Timedelta(minutes=90)) == 6
    assert candles_elapsed_m15(_T0, _T0) == 0


# -- salida temporal post-entrada: 6 velas / 90 min sin +0.5R ----------------

def test_temporal_exit_not_triggered_before_6_candles_even_without_progress():
    current = _T0 + pd.Timedelta(minutes=75)  # 5 candles
    assert temporal_exit_triggered(_T0, current, progress_r=0.0) is False


def test_temporal_exit_triggered_at_exactly_6_candles_without_0_5r():
    current = _T0 + pd.Timedelta(minutes=90)  # exactly 6 candles
    assert temporal_exit_triggered(_T0, current, progress_r=0.4) is True


def test_temporal_exit_not_triggered_at_6_candles_if_0_5r_reached():
    current = _T0 + pd.Timedelta(minutes=90)
    assert temporal_exit_triggered(_T0, current, progress_r=0.5) is False


def test_temporal_exit_not_triggered_past_6_candles_if_progress_reached():
    current = _T0 + pd.Timedelta(minutes=120)  # 8 candles, well past 6
    assert temporal_exit_triggered(_T0, current, progress_r=0.6) is False


def test_temporal_exit_triggered_well_past_6_candles_without_progress():
    current = _T0 + pd.Timedelta(minutes=180)
    assert temporal_exit_triggered(_T0, current, progress_r=0.2) is True


# -- expiración del setup si no retrocede al 50% (mismo umbral) -------------

def test_fvg_not_expired_before_6_candles():
    current = _T0 + pd.Timedelta(minutes=75)
    assert fvg_setup_expired(_T0, current, retraced_to_50pct=False) is False


def test_fvg_expires_at_exactly_6_candles_without_retracement():
    current = _T0 + pd.Timedelta(minutes=90)
    assert fvg_setup_expired(_T0, current, retraced_to_50pct=False) is True


def test_fvg_never_expires_once_retraced_regardless_of_time():
    current = _T0 + pd.Timedelta(minutes=300)  # way past the threshold
    assert fvg_setup_expired(_T0, current, retraced_to_50pct=True) is False
