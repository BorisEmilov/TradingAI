"""Chequeo de causalidad para mtf_strategies/: ningún componente nuevo debe
usar datos posteriores al punto de decisión. Mismo principio que el resto del
proyecto (ver TimeframeAnalysis.as_of()) -- construir el análisis con MÁS
datos futuros que el cutoff nunca debe cambiar lo que se reporta EN ese
cutoff. Usa datos sintéticos deterministas (no depende de nada cacheado en
disco) para que corra rápido y sin datos externos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trader.config import load_config
from trader.mtf_strategies.analysis import PrecomputedHistory, build_analysis
from trader.mtf_strategies.continuation import evaluate_continuation
from trader.mtf_strategies.reversal import evaluate_reversal


def _synthetic_series(n: int, start: str, freq: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    high = close + np.abs(rng.normal(0.1, 0.05, n))
    low = close - np.abs(rng.normal(0.1, 0.05, n))
    open_ = close + rng.normal(0, 0.05, n)
    volume = np.full(n, 1000.0)
    spread = np.full(n, 2.0)
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume, "spread": spread})


@pytest.fixture
def config():
    return load_config()


def test_precomputed_history_as_of_unaffected_by_appending_future_rows(config):
    base = _synthetic_series(400, "2024-01-01", "1h", seed=1)
    cutoff = base["timestamp"].iloc[250]

    short = base.iloc[:300].reset_index(drop=True)  # ends shortly after cutoff
    # extended: same history up to cutoff, plus a LOT more (adversarial: a
    # sharp synthetic crash) appended afterward
    crash = pd.DataFrame({
        "timestamp": pd.date_range(base["timestamp"].iloc[300], periods=200, freq="1h", tz="UTC"),
        "open": np.linspace(120, 40, 200), "high": np.linspace(121, 41, 200),
        "low": np.linspace(119, 39, 200), "close": np.linspace(120, 40, 200),
        "volume": 1000.0, "spread": 2.0,
    })
    extended = pd.concat([short, crash], ignore_index=True)

    pre_short = PrecomputedHistory(short, "H1", config)
    pre_extended = PrecomputedHistory(extended, "H1", config)

    a = pre_short.as_of(cutoff)
    b = pre_extended.as_of(cutoff)

    assert len(a.df) == len(b.df)
    assert [(s.kind, s.price, s.timestamp) for s in a.swings] == [(s.kind, s.price, s.timestamp) for s in b.swings]
    assert [(s.kind, s.direction, s.timestamp) for s in a.structure_events] == [(s.kind, s.direction, s.timestamp) for s in b.structure_events]
    assert [(o.timestamp, o.confirmed_at, o.broken_at) for o in a.order_blocks] == [(o.timestamp, o.confirmed_at, o.broken_at) for o in b.order_blocks]
    assert [(i.timestamp, i.direction) for i in a.inverted_order_blocks] == [(i.timestamp, i.direction) for i in b.inverted_order_blocks]
    assert [(f.timestamp, f.direction) for f in a.fvgs] == [(f.timestamp, f.direction) for f in b.fvgs]


def test_precomputed_history_matches_recompute_from_scratch_at_a_cutoff_before_a_future_crash(config):
    """PrecomputedHistory.as_of() debe dar lo mismo que recomputar desde cero
    sobre datos truncados en el momento -- incluso cuando el futuro (no
    visible en el cutoff) contiene algo tan extremo como un crash sintético."""
    base = _synthetic_series(300, "2024-01-01", "1h", seed=2)
    crash = pd.DataFrame({
        "timestamp": pd.date_range(base["timestamp"].iloc[-1] + pd.Timedelta(hours=1), periods=150, freq="1h", tz="UTC"),
        "open": np.linspace(115, 30, 150), "high": np.linspace(116, 31, 150),
        "low": np.linspace(114, 29, 150), "close": np.linspace(115, 30, 150),
        "volume": 1000.0, "spread": 2.0,
    })
    full = pd.concat([base, crash], ignore_index=True)
    cutoff = base["timestamp"].iloc[200]

    pre = PrecomputedHistory(full, "H1", config)
    from_precomputed = pre.as_of(cutoff)
    from_scratch = build_analysis(full, "H1", cutoff, config)

    assert len(from_precomputed.df) == len(from_scratch.df)
    assert from_precomputed.df["timestamp"].iloc[-1] == from_scratch.df["timestamp"].iloc[-1]
    assert [(s.kind, s.price) for s in from_precomputed.swings] == [(s.kind, s.price) for s in from_scratch.swings]
    assert [(o.confirmed_at, o.broken_at) for o in from_precomputed.order_blocks] == [(o.confirmed_at, o.broken_at) for o in from_scratch.order_blocks]


def test_evaluate_continuation_and_reversal_unaffected_by_future_data(config):
    """Extremo a extremo: correr las 2 estrategias en un cutoff da el MISMO
    resultado sin importar si el dataset de entrada tiene 50 velas más allá
    de ese cutoff o 500 -- ninguna de las dos debe filtrar futuro."""
    h1_base = _synthetic_series(500, "2024-01-01", "1h", seed=3)
    m15_base = _synthetic_series(2000, "2024-01-01", "15min", seed=4)
    d1_base = _synthetic_series(60, "2023-10-01", "1D", seed=5)
    h4_base = (
        h1_base.set_index("timestamp").resample("4h", origin="start_day")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index()
    )

    cutoff = m15_base["timestamp"].iloc[1500]

    def _extend(df: pd.DataFrame, freq: str, n_extra: int) -> pd.DataFrame:
        extra = pd.date_range(df["timestamp"].iloc[-1] + pd.Timedelta(freq), periods=n_extra, freq=freq, tz="UTC")
        rng = np.random.default_rng(99)
        close = df["close"].iloc[-1] + np.cumsum(rng.normal(0, 0.5, n_extra))
        extra_df = pd.DataFrame({
            "timestamp": extra, "open": close, "high": close + 0.2, "low": close - 0.2, "close": close,
            "volume": 1000.0, "spread": 2.0,
        })
        return pd.concat([df, extra_df], ignore_index=True)

    m15_extended = _extend(m15_base, "15min", 500)
    h1_extended = _extend(h1_base, "1h", 200)
    h4_extended = _extend(h4_base, "4h", 50)
    d1_extended = _extend(d1_base, "1D", 30)

    from trader.events import closed_candles_as_of

    def _summary(r):
        if isinstance(r, list):
            return [(s.direction, s.entry, s.sl, s.tp2) for s in r]
        return (r.reason, r.stage)

    results = []
    for h4_df, h1_df, m15_df, d1_df in [
        (h4_base, h1_base, m15_base, d1_base),
        (h4_extended, h1_extended, m15_extended, d1_extended),
    ]:
        h4 = PrecomputedHistory(h4_df, "H4", config).as_of(cutoff)
        h1 = PrecomputedHistory(h1_df, "H1", config).as_of(cutoff)
        m15 = PrecomputedHistory(m15_df, "M15", config).as_of(cutoff)
        d1_as_of = closed_candles_as_of(d1_df, "D1", cutoff)
        price = float(m15.df["close"].iloc[-1])

        cont = evaluate_continuation("SYN", h4, h1, m15, d1_as_of, config, cutoff, price)
        rev = evaluate_reversal("SYN", h4, h1, m15, d1_as_of, config, cutoff)
        results.append((_summary(cont), _summary(rev)))

    assert results[0] == results[1]  # short-history vs future-extended -- identical at the same cutoff
