from __future__ import annotations

import pandas as pd
import pytest

from trader.mtf_strategies.session_risk import (
    DailyLossState,
    PositionConcurrencyState,
    active_entry_session,
    can_open_new_trade,
    compute_position_size_lots,
    daily_loss_limit_reached,
    entry_session_end,
    record_daily_result,
    record_position_closed,
    record_position_opened,
)


# -- ventanas DST-aware ------------------------------------------------------

def test_london_window_winter_gmt_no_offset():
    # London local = UTC in winter (GMT, no DST) -- 09:00 UTC is 09:00 local, inside 08-11
    assert active_entry_session(pd.Timestamp("2024-01-15 09:00", tz="UTC")) == "london"


def test_london_window_summer_bst_one_hour_offset():
    # British Summer Time: local = UTC+1 -- 08:30 local is 07:30 UTC, inside the 08-11 LOCAL window
    assert active_entry_session(pd.Timestamp("2024-07-15 07:30", tz="UTC")) == "london"
    # 07:00 UTC = 08:00 BST -- just inside; 06:59 UTC = 07:59 BST -- just outside
    assert active_entry_session(pd.Timestamp("2024-07-15 06:59", tz="UTC")) != "london"


def test_new_york_window_winter_est_five_hour_offset():
    # EST = UTC-5 in winter -- 08:00 local = 13:00 UTC
    assert active_entry_session(pd.Timestamp("2024-01-15 13:00", tz="UTC")) == "new_york"


def test_new_york_window_summer_edt_four_hour_offset():
    # EDT = UTC-4 in summer -- 08:30 local = 12:30 UTC
    assert active_entry_session(pd.Timestamp("2024-07-15 12:30", tz="UTC")) == "new_york"


def test_outside_both_windows_is_none():
    assert active_entry_session(pd.Timestamp("2024-01-15 18:00", tz="UTC")) is None


# -- entry_session_end (Prompt 2026-09-23: expiración de orden límite = fin de sesión) --

def test_entry_session_end_london_winter_gmt():
    end = entry_session_end("london", pd.Timestamp("2024-01-15 09:00", tz="UTC"))
    assert end == pd.Timestamp("2024-01-15 11:00", tz="UTC")  # GMT, sin offset


def test_entry_session_end_london_summer_bst():
    end = entry_session_end("london", pd.Timestamp("2024-07-15 07:30", tz="UTC"))
    assert end == pd.Timestamp("2024-07-15 10:00", tz="UTC")  # BST = UTC+1 -> 11:00 local = 10:00 UTC


def test_entry_session_end_new_york_winter_est():
    end = entry_session_end("new_york", pd.Timestamp("2024-01-15 13:00", tz="UTC"))
    assert end == pd.Timestamp("2024-01-15 16:00", tz="UTC")  # EST = UTC-5 -> 11:00 local = 16:00 UTC


def test_entry_session_end_new_york_summer_edt():
    end = entry_session_end("new_york", pd.Timestamp("2024-07-15 12:30", tz="UTC"))
    assert end == pd.Timestamp("2024-07-15 15:00", tz="UTC")  # EDT = UTC-4 -> 11:00 local = 15:00 UTC


def test_entry_session_end_is_always_after_generated_at():
    # generated_at cerca del cierre (10:55 local, GMT) -- ventana corta pero
    # siempre positiva, sin mínimo artificial (instrucción 2 del prompt)
    generated_at = pd.Timestamp("2024-01-15 10:55", tz="UTC")
    end = entry_session_end("london", generated_at)
    assert end > generated_at
    assert (end - generated_at) == pd.Timedelta(minutes=5)


# -- concurrencia POR PAR (corrección Prompt 5 -- no un límite global de sesión) --

def test_two_different_symbols_qualifying_in_the_same_session_both_execute():
    """Corrección del test del prompt anterior: el límite NO es global entre
    símbolos -- EURUSD y GBPUSD pueden tener posiciones simultáneas."""
    state = PositionConcurrencyState()

    ok_eur, reason_eur = can_open_new_trade(state, "EURUSD")
    assert ok_eur is True and reason_eur is None
    state = record_position_opened(state, "EURUSD")

    ok_gbp, reason_gbp = can_open_new_trade(state, "GBPUSD")
    assert ok_gbp is True and reason_gbp is None  # otro símbolo, sin conflicto


def test_same_symbol_fires_twice_while_first_position_still_open_second_is_discarded():
    """El caso correcto: NO es por sesión, es por símbolo -- una segunda señal
    en el MISMO par mientras la primera posición sigue abierta se descarta,
    sin importar si sigue siendo la misma sesión o no."""
    state = PositionConcurrencyState()
    state = record_position_opened(state, "EURUSD")

    ok, reason = can_open_new_trade(state, "EURUSD")

    assert ok is False
    assert reason == "position_already_open_in_EURUSD"


def test_symbol_becomes_available_again_once_its_position_closes():
    state = PositionConcurrencyState()
    state = record_position_opened(state, "EURUSD")
    state = record_position_closed(state, "EURUSD")

    ok, reason = can_open_new_trade(state, "EURUSD")

    assert ok is True and reason is None


def test_different_symbol_never_blocked_by_another_symbols_open_position():
    state = PositionConcurrencyState()
    state = record_position_opened(state, "USDJPY")

    ok, reason = can_open_new_trade(state, "EURUSD")

    assert ok is True and reason is None


# -- corte de pérdida diaria -1.5R (global, ahora separado del tracking por símbolo) --

def test_daily_loss_limit_not_yet_reached_stays_open():
    state = DailyLossState(date=pd.Timestamp("2024-01-15", tz="UTC"))
    state = record_daily_result(state, pd.Timestamp("2024-01-15 09:00", tz="UTC"), r_multiple=-1.0)
    assert state.locked_out is False
    _, reached = daily_loss_limit_reached(state, pd.Timestamp("2024-01-15 13:00", tz="UTC"))
    assert reached is False


def test_daily_loss_limit_exactly_at_threshold_locks_out():
    state = DailyLossState(date=pd.Timestamp("2024-01-15", tz="UTC"))
    state = record_daily_result(state, pd.Timestamp("2024-01-15 09:00", tz="UTC"), r_multiple=-1.0)
    state = record_daily_result(state, pd.Timestamp("2024-01-15 13:00", tz="UTC"), r_multiple=-0.5)
    assert state.cumulative_r == -1.5
    assert state.locked_out is True
    _, reached = daily_loss_limit_reached(state, pd.Timestamp("2024-01-15 14:00", tz="UTC"))
    assert reached is True


def test_daily_loss_limit_resets_on_a_new_day():
    state = DailyLossState(date=pd.Timestamp("2024-01-15", tz="UTC"))
    state = record_daily_result(state, pd.Timestamp("2024-01-15 09:00", tz="UTC"), r_multiple=-2.0)
    assert state.locked_out is True
    state, reached = daily_loss_limit_reached(state, pd.Timestamp("2024-01-16 09:00", tz="UTC"))
    assert reached is False  # nuevo dia, el estado se resetea


def test_daily_loss_lockout_never_touches_an_already_open_position():
    """Confirmación explícita (Prompt 6): al llegar a -1.5R se BLOQUEAN
    entradas nuevas -- las posiciones ya abiertas NO se cierran de golpe.
    `DailyLossState` y `PositionConcurrencyState` son independientes por
    construcción: activar el lockout no muta ni consulta el estado de
    concurrencia en absoluto."""
    concurrency = PositionConcurrencyState()
    concurrency = record_position_opened(concurrency, "EURUSD")  # posición abierta ANTES del lockout

    daily = DailyLossState(date=pd.Timestamp("2024-01-15", tz="UTC"))
    daily = record_daily_result(daily, pd.Timestamp("2024-01-15 09:30", tz="UTC"), r_multiple=-1.5)
    assert daily.locked_out is True

    # la posición de EURUSD sigue figurando como abierta -- nada la tocó
    assert "EURUSD" in concurrency.open_symbols

    # una entrada NUEVA sí queda bloqueada -- pero por el chequeo de lockout,
    # que el llamador tiene que consultar aparte (no hay una función única
    # que combine ambos, a propósito: son 2 gates independientes)
    _, reached = daily_loss_limit_reached(daily, pd.Timestamp("2024-01-15 10:00", tz="UTC"))
    assert reached is True


def test_daily_loss_limit_is_independent_of_which_symbol_traded():
    """La pérdida diaria sigue siendo GLOBAL -- se acumula sin importar en
    qué símbolo ocurrió cada operación (a diferencia de la concurrencia, que
    sí es por símbolo)."""
    state = DailyLossState(date=pd.Timestamp("2024-01-15", tz="UTC"))
    state = record_daily_result(state, pd.Timestamp("2024-01-15 09:00", tz="UTC"), r_multiple=-0.8)  # EURUSD, say
    state = record_daily_result(state, pd.Timestamp("2024-01-15 09:05", tz="UTC"), r_multiple=-0.8)  # GBPUSD, say
    assert state.cumulative_r == pytest.approx(-1.6)
    assert state.locked_out is True


# -- sizing por % de equity --------------------------------------------------

def test_position_size_respects_risk_budget():
    lots = compute_position_size_lots(
        equity=10_000, risk_pct=0.0025, entry=1.1000, sl=1.0950,
        trade_tick_value=1.0, trade_tick_size=0.00001, volume_min=0.01, volume_max=500, volume_step=0.01,
    )
    risk_price = 0.0050
    actual_risk_usd = lots * (risk_price / 0.00001) * 1.0
    assert actual_risk_usd <= 10_000 * 0.0025  # never MORE than budgeted (rounds down)
    assert lots > 0


def test_position_size_below_minimum_lot_returns_zero():
    lots = compute_position_size_lots(
        equity=10, risk_pct=0.0025, entry=1.1000, sl=1.0950,  # tiny equity
        trade_tick_value=1.0, trade_tick_size=0.00001, volume_min=0.01, volume_max=500, volume_step=0.01,
    )
    assert lots == 0.0


def test_position_size_rejects_risk_pct_outside_allowed_range():
    with pytest.raises(ValueError):
        compute_position_size_lots(
            equity=10_000, risk_pct=0.01, entry=1.1, sl=1.09,  # 1% -- outside 0.25-0.50%
            trade_tick_value=1.0, trade_tick_size=0.00001, volume_min=0.01, volume_max=500, volume_step=0.01,
        )
