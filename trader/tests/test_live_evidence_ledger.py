"""Criterio de abandono pre-registrado del piloto (logs/live_pilot_prereg_2026-10-02.md)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import live_evidence_ledger as led  # noqa: E402


def _series(rs, per_day=1):
    days = pd.date_range("2026-10-05", periods=len(rs), freq="D", tz="UTC")
    return pd.DataFrame({"net_r": rs, "day": days[np.arange(len(rs)) // per_day]})


def test_no_abandonment_below_20_trades_even_if_terrible():
    assert led.evaluate(_series([-1.0] * 19))["abandono"] == []


def test_abandon_when_mean_below_minus_0_3_with_20_trades():
    ev = led.evaluate(_series([-1.0] * 14 + [1.0] * 6))  # media -0.4
    assert any("R medio" in t for t in ev["abandono"])


def test_abandon_when_ci90_entirely_negative():
    # media -0.25 (no dispara el umbral de media) pero sin dispersión suficiente para tocar 0
    ev = led.evaluate(_series([-0.3, -0.2] * 10))
    assert ev["r_medio"] > led.MAX_MEAN_R and any("IC90" in t for t in ev["abandono"])


def test_mixed_series_continues_and_flags_n50():
    ev = led.evaluate(_series([1.5, -1.0] * 25, per_day=2))
    assert ev["abandono"] == [] and ev["reevaluar_n50"] is True
