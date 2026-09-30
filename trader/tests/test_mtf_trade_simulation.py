from __future__ import annotations

from dataclasses import replace

import pandas as pd

from trader.mtf_strategies.signal import MTFSignal
from trader.mtf_strategies.trade_simulation import (
    simulate_pending_order, simulate_pending_order_outcome, simulate_position_close,
)

_STEP = pd.Timedelta(minutes=15)


def _signal(entry, sl, tp1, tp2, direction, fvg_ts) -> MTFSignal:
    return MTFSignal(
        strategy="continuation", setup=None, symbol="TEST", direction=direction, session="london",
        entry=entry, sl=sl, tp1=tp1, tp2=tp2, risk_reward=3.0,
        sweep_timestamp=fvg_ts, mss_timestamp=fvg_ts, fvg_confirmed_at=fvg_ts, generated_at=fvg_ts,
    )


def test_long_reaches_tp1_then_tp2_closes_at_tp2_touch():
    ts = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "open":  [101, 100, 99, 100, 101, 103, 104, 105, 106, 107],
        "high":  [101, 100.5, 99.5, 101, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5],
        "low":   [100.5, 99.5, 98.5, 99.5, 100.5, 102.5, 103.5, 104.5, 105.5, 106.5],
        "close": [100.8, 99.8, 99, 100.5, 102, 103, 104, 105, 106, 107],
    })
    sig = _signal(100.0, 98.0, 102.0, 106.0, "long", ts[0])

    close_ts = simulate_position_close(sig, df)

    assert close_ts == ts[8] + _STEP  # tp2 (106) first touched at bar 8's high (106.5)


def test_short_hits_sl_before_any_target():
    ts = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "open":  [100, 100.5, 101, 102, 103, 104, 105, 106, 107, 108],
        "high":  [100.2, 100.8, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5],
        "low":   [99.8, 100.2, 100.8, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5],
        "close": [100, 100.5, 101, 102, 103, 104, 105, 106, 107, 108],
    })
    sig = _signal(100.0, 102.0, 98.0, 94.0, "short", ts[0])

    close_ts = simulate_position_close(sig, df)

    assert close_ts == ts[3] + _STEP  # sl (102) first touched at bar 3's high (102.5)


def test_temporal_exit_fires_at_exactly_6_candles_without_0_5r_progress():
    ts = pd.date_range("2024-01-01", periods=12, freq="15min", tz="UTC")
    flat = pd.DataFrame({"timestamp": ts, "open": 100.1, "high": 100.3, "low": 99.9, "close": 100.1})
    sig = _signal(100.0, 98.0, 102.0, 106.0, "long", ts[0])

    close_ts = simulate_position_close(sig, flat)

    entry_time = ts[0] + _STEP  # first bar (index 0) already touches entry (low=99.9<=100)
    assert close_ts == entry_time + 6 * _STEP


def test_sl_and_tp1_touched_same_bar_sl_wins_conservative_convention():
    ts = pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "open":  [100, 99.5, 99.5, 99.5, 99.5],
        "high":  [100.2, 100.5, 102.5, 103, 103],  # bar 2: high touches tp1 (102) AND low touches sl (98)
        "low":   [99.8, 99, 97.5, 97, 97],
        "close": [100, 99.2, 100, 100, 100],
    })
    sig = _signal(100.0, 98.0, 102.0, 106.0, "long", ts[0])

    close_ts = simulate_position_close(sig, df)

    assert close_ts == ts[2] + _STEP  # SL wins even though TP1 was ALSO touched in the same bar


def test_returns_none_when_entry_never_touched():
    # entry=100 is a pullback TARGET below current price -- a resting limit
    # buy at 100 only fills once price actually comes DOWN to it. Price
    # staying entirely ABOVE 100 (110-111) never fills it. (Price staying
    # entirely BELOW 100 would instead mean the limit was already marketable
    # from the first bar -- a different case, not "never touched".)
    ts = pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "open": 110, "high": 111, "low": 109, "close": 110})
    sig = _signal(100.0, 98.0, 102.0, 106.0, "long", ts[0])

    assert simulate_position_close(sig, df) is None


def test_returns_none_when_data_runs_out_before_resolution():
    ts = pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "open": 100, "high": 100.5, "low": 99.5, "close": 100})  # touches entry, never resolves
    sig = _signal(100.0, 98.0, 102.0, 106.0, "long", ts[0])

    assert simulate_position_close(sig, df) is None


def test_pending_order_fills_and_closes_when_touched_inside_window():
    # entry se toca en la barra 2 (low=99.8<=100), a 30 min de fvg_confirmed_at
    # -- dentro de una ventana de 90 min. Debe llenarse y seguir simulando
    # hasta el cierre real (mismo motor que simulate_position_close).
    ts = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "open":  [101, 101, 99.9, 100, 101, 103, 104, 105, 106, 107],
        "high":  [101, 101, 100.2, 101, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5],
        "low":   [100.5, 100.5, 99.8, 99.5, 100.5, 102.5, 103.5, 104.5, 105.5, 106.5],
        "close": [100.8, 100.8, 100, 100.5, 102, 103, 104, 105, 106, 107],
    })
    sig = _signal(100.0, 98.0, 102.0, 106.0, "long", ts[0])

    filled, closed_at = simulate_pending_order(sig, df, deadline=ts[0] + pd.Timedelta(minutes=90))

    assert filled is True
    assert closed_at == ts[8] + _STEP  # tp2 (106) first touched at bar 8's high (106.5), same as simulate_position_close


def test_pending_order_expires_when_never_touched_inside_window():
    # el precio se queda lejos del nivel de entrada durante toda la ventana --
    # la orden vence sin llenarse, el símbolo se libera exactamente en el
    # deadline dado (quién lo calculó -- offset fijo o fin de sesión -- es
    # responsabilidad del llamador, no de esta función).
    ts = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "open": 110, "high": 111, "low": 109, "close": 110})
    sig = _signal(100.0, 98.0, 102.0, 115.0, "long", ts[0])
    deadline = ts[0] + pd.Timedelta(minutes=90)

    filled, freed_at = simulate_pending_order(sig, df, deadline=deadline)

    assert filled is False
    assert freed_at == deadline


def test_pending_order_touch_after_window_does_not_count_as_fill():
    # el precio SÍ vuelve al nivel de entrada eventualmente, pero recién en la
    # barra 7 (105 min desde fvg_confirmed_at) -- fuera de la ventana de 90
    # min (6 velas). Bajo el modelo de orden límite real, ya expiró antes.
    ts = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
    high = [111, 111, 111, 111, 111, 111, 111, 100.2, 111, 111]
    low = [109, 109, 109, 109, 109, 109, 109, 99.8, 109, 109]
    df = pd.DataFrame({"timestamp": ts, "open": 110, "high": high, "low": low, "close": 110})
    sig = _signal(100.0, 98.0, 102.0, 115.0, "long", ts[0])

    filled, freed_at = simulate_pending_order(sig, df, deadline=ts[0] + pd.Timedelta(minutes=90))

    assert filled is False
    assert freed_at == ts[0] + pd.Timedelta(minutes=90)


def test_outcome_r_partial_then_tp2_and_straight_sl():
    # mismo escenario que el test de fill: TP1 (+1R) parcial 50% y resto a TP2 (+3R) -> +2R.
    ts = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "open":  [101, 101, 99.9, 100, 101, 103, 104, 105, 106, 107],
        "high":  [101, 101, 100.2, 101, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5],
        "low":   [100.5, 100.5, 99.8, 99.5, 100.5, 102.5, 103.5, 104.5, 105.5, 106.5],
        "close": [100.8, 100.8, 100, 100.5, 102, 103, 104, 105, 106, 107],
    })
    sig = _signal(100.0, 98.0, 102.0, 106.0, "long", ts[0])
    filled, closed_at, entry_time, r = simulate_pending_order_outcome(sig, df, deadline=ts[0] + pd.Timedelta(minutes=90))
    assert filled and entry_time == ts[2] + _STEP and closed_at == ts[8] + _STEP
    assert abs(r - 2.0) < 1e-9

    # SL directo sin parcial -> exactamente -1R
    crash = df.assign(low=[100.5, 100.5, 99.8, 97.0, 97, 97, 97, 97, 97, 97])
    _, _, _, r_sl = simulate_pending_order_outcome(sig, crash, deadline=ts[0] + pd.Timedelta(minutes=90))
    assert r_sl == -1.0


# -- alineación con el piloto (2026-09-29): colocación en generated_at, solo si el precio no cruzó la entrada --

def test_pending_order_not_placed_when_price_already_past_entry():
    # short con entrada 100: al colocar el mercado ya está en 101 (por encima) -> MT5 la rechazaría
    ts = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "open": 101, "high": 101.5, "low": 99.5, "close": 101})
    sig = _signal(100.0, 102.0, 98.0, 94.0, "short", ts[0])

    filled, freed_at, entry_time, r = simulate_pending_order_outcome(sig, df, deadline=ts[-1])

    assert (filled, freed_at, entry_time, r) == (False, ts[0], None, None)


def test_touch_before_placement_does_not_count_as_fill():
    # el precio toca la entrada en la vela 0 (entre FVG y generated_at), la orden recién existe desde ts[2]
    ts = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
    low = [99.8, 101, 101, 101, 101, 101, 101, 101, 101, 101]
    df = pd.DataFrame({"timestamp": ts, "open": 101, "high": 101.5, "low": low, "close": 101})
    sig = replace(_signal(100.0, 98.0, 102.0, 106.0, "long", ts[0]), generated_at=ts[2])

    filled, _, _, _ = simulate_pending_order_outcome(sig, df, deadline=ts[-1])

    assert filled is False
