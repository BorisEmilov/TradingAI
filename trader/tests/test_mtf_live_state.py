from __future__ import annotations

import pandas as pd
import pytest

from trader.mtf_strategies.live_state import (
    ExitDeal,
    concurrency_open_symbols,
    exit_deals_from_history,
    realized_r_from_deals,
    round_down_to_step,
    signal_key,
    tp1_touched,
)
from trader.mtf_strategies.signal import MTFSignal


def _signal(**overrides) -> MTFSignal:
    base = dict(
        strategy="continuation", setup=None, symbol="EURUSD", direction="long", session="london",
        entry=1.1, sl=1.09, tp1=1.11, tp2=1.13, risk_reward=3.0,
        sweep_timestamp=pd.Timestamp("2024-01-01 08:00", tz="UTC"),
        mss_timestamp=pd.Timestamp("2024-01-01 09:00", tz="UTC"),
        fvg_confirmed_at=pd.Timestamp("2024-01-01 09:15", tz="UTC"),
        generated_at=pd.Timestamp("2024-01-01 09:30", tz="UTC"),
    )
    base.update(overrides)
    return MTFSignal(**base)


def test_signal_key_identical_for_the_same_underlying_setup_regenerated_later():
    first = _signal(generated_at=pd.Timestamp("2024-01-01 09:30", tz="UTC"))
    later = _signal(generated_at=pd.Timestamp("2024-01-01 09:45", tz="UTC"))  # re-detected next poll
    assert signal_key(first) == signal_key(later)


def test_signal_key_differs_for_a_different_setup():
    a = _signal(sweep_timestamp=pd.Timestamp("2024-01-01 08:00", tz="UTC"))
    b = _signal(sweep_timestamp=pd.Timestamp("2024-01-02 08:00", tz="UTC"))
    assert signal_key(a) != signal_key(b)


def test_tp1_touched_long_and_short():
    assert tp1_touched("long", bar_high=1.12, bar_low=1.10, tp1=1.11) is True
    assert tp1_touched("long", bar_high=1.10, bar_low=1.09, tp1=1.11) is False
    assert tp1_touched("short", bar_high=1.10, bar_low=1.089, tp1=1.09) is True
    assert tp1_touched("short", bar_high=1.10, bar_low=1.095, tp1=1.09) is False


def test_round_down_to_step_never_rounds_up():
    assert round_down_to_step(0.37, 0.01) == pytest_approx(0.37)
    assert round_down_to_step(0.105, 0.01) == pytest_approx(0.10)
    assert round_down_to_step(0.0, 0.01) == 0.0


def pytest_approx(x):
    return pytest.approx(x, abs=1e-9)


def test_realized_r_full_close_at_tp1_level_is_positive_one_r():
    deals = [ExitDeal(volume=0.1, price=1.11, entry_name="OUT", reason_name="TP")]
    r = realized_r_from_deals("long", entry_price=1.10, initial_sl=1.09, exit_deals=deals)
    assert r == pytest_approx(1.0)


def test_realized_r_full_close_at_sl_is_minus_one_r_long():
    deals = [ExitDeal(volume=0.1, price=1.09, entry_name="OUT", reason_name="SL")]
    r = realized_r_from_deals("long", entry_price=1.10, initial_sl=1.09, exit_deals=deals)
    assert r == pytest_approx(-1.0)


def test_realized_r_full_close_at_sl_is_minus_one_r_short():
    deals = [ExitDeal(volume=0.1, price=1.11, entry_name="OUT", reason_name="SL")]
    r = realized_r_from_deals("short", entry_price=1.10, initial_sl=1.11, exit_deals=deals)
    assert r == pytest_approx(-1.0)


def test_realized_r_weights_partial_then_final_leg_by_volume():
    # 50% closed at TP1 (+1R), remaining 50% closed later at breakeven (0R) -> overall +0.5R
    deals = [
        ExitDeal(volume=0.05, price=1.11, entry_name="OUT", reason_name="CLIENT"),  # partial at tp1 (+1R)
        ExitDeal(volume=0.05, price=1.10, entry_name="OUT_BY", reason_name="SL"),   # remainder at breakeven (0R)
    ]
    r = realized_r_from_deals("long", entry_price=1.10, initial_sl=1.09, exit_deals=deals)
    assert r == pytest_approx(0.5)


def test_realized_r_ignores_entry_deals():
    deals_raw = [
        {"volume": 0.1, "price": 1.10, "entry_name": "IN", "reason_name": "CLIENT"},
        {"volume": 0.1, "price": 1.13, "entry_name": "OUT", "reason_name": "TP"},
    ]
    exits = exit_deals_from_history(deals_raw)
    assert len(exits) == 1
    assert exits[0].price == 1.13
    r = realized_r_from_deals("long", entry_price=1.10, initial_sl=1.09, exit_deals=exits)
    assert r == pytest_approx(3.0)


def test_realized_r_no_exit_deals_is_zero():
    assert realized_r_from_deals("long", entry_price=1.10, initial_sl=1.09, exit_deals=[]) == 0.0


def test_concurrency_open_symbols_derives_from_tracked_positions():
    open_positions = {
        "111": {"symbol": "EURUSD"},
        "222": {"symbol": "GBPUSD"},
    }
    assert concurrency_open_symbols(open_positions) == {"EURUSD", "GBPUSD"}


def test_concurrency_open_symbols_empty_when_nothing_tracked():
    assert concurrency_open_symbols({}) == set()
