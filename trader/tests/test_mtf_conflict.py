from __future__ import annotations

import pandas as pd

from trader.mtf_strategies.conflict import resolve
from trader.mtf_strategies.signal import MTFSignal, NoSignal


def _signal(strategy: str, symbol: str = "EURUSD", session: str = "london") -> MTFSignal:
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    return MTFSignal(
        strategy=strategy, setup=None, symbol=symbol, direction="long", session=session,
        entry=1.10, sl=1.09, tp1=1.11, tp2=1.13, risk_reward=3.0,
        sweep_timestamp=ts, mss_timestamp=ts, fvg_confirmed_at=ts, generated_at=ts,
    )


def test_both_strategies_signal_same_symbol_and_session_discards_both_and_logs_conflict():
    cont = [_signal("continuation")]
    rev = [_signal("reversal")]

    survivors, conflicts = resolve(cont, rev)

    assert survivors == []
    assert len(conflicts) == 1
    assert conflicts[0].symbol == "EURUSD"
    assert conflicts[0].session == "london"
    assert conflicts[0].continuation_signal.strategy == "continuation"
    assert conflicts[0].reversal_signal.strategy == "reversal"


def test_only_continuation_signals_passes_through_no_conflict():
    survivors, conflicts = resolve([_signal("continuation")], NoSignal(strategy="reversal", reason="x", stage="y"))
    assert len(survivors) == 1
    assert conflicts == []


def test_only_reversal_signals_passes_through_no_conflict():
    survivors, conflicts = resolve(NoSignal(strategy="continuation", reason="x", stage="y"), [_signal("reversal")])
    assert len(survivors) == 1
    assert conflicts == []


def test_neither_signals_no_conflict_no_survivors():
    survivors, conflicts = resolve(
        NoSignal(strategy="continuation", reason="x", stage="y"), NoSignal(strategy="reversal", reason="x", stage="y")
    )
    assert survivors == [] and conflicts == []


def test_different_symbols_no_conflict_both_survive():
    survivors, conflicts = resolve([_signal("continuation", symbol="EURUSD")], [_signal("reversal", symbol="GBPUSD")])
    assert len(survivors) == 2
    assert conflicts == []


def test_different_sessions_same_symbol_is_still_a_conflict():
    # Corrección Prompt 5: el conflicto está scopeado por SÍMBOLO, no por
    # símbolo+sesión -- antes esto pasaba sin conflicto porque la sesión no
    # coincidía, remanente del modelo de concurrencia global ya descartado.
    survivors, conflicts = resolve(
        [_signal("continuation", session="london")], [_signal("reversal", session="new_york")]
    )
    assert survivors == []
    assert len(conflicts) == 1


def test_non_conflicting_signal_survives_alongside_a_conflict():
    cont = [_signal("continuation", symbol="EURUSD"), _signal("continuation", symbol="GBPUSD")]
    rev = [_signal("reversal", symbol="EURUSD")]  # conflicts only with the EURUSD continuation signal

    survivors, conflicts = resolve(cont, rev)

    assert len(conflicts) == 1
    assert len(survivors) == 1
    assert survivors[0].symbol == "GBPUSD"
