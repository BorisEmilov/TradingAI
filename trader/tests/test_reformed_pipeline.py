"""End-to-end smoke test for the reformulated 4-layer orchestration
(trader/pipeline/reformed.py): runs the full D1->dominant reason->M15
confirmation->management pipeline over a synthetic random-walk dataset large
enough to clear every minimum-sample requirement (30+ events for the
expanding percentiles, 500+ M15 candles for the rolling confirmation
window), and checks it (a) never crashes and (b) any signals it DOES produce
satisfy the design's own invariants. It intentionally does not assert a
specific trade count -- a synthetic random walk isn't expected to produce
real structure, and zero signals is itself a valid, informative outcome
consistent with every prior phase of this project.
"""

import numpy as np
import pandas as pd

from trader.management import BASE_RISK_PCT, CONVICTION_MAX, CONVICTION_MIN
from trader.pipeline.engine import TimeframeAnalysis
from trader.pipeline.reformed import generate_reformed_signals
from tests.test_pipeline_engine import _toy_config, _always_on_sessions


def _random_walk_candles(n: int, freq: str, seed: int, start_price: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start=pd.Timestamp("2024-01-01", tz="UTC"), periods=n, freq=freq)
    closes = start_price + np.cumsum(rng.normal(0, 0.3, size=n))
    opens = np.concatenate([[start_price], closes[:-1]])
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, 0.15, size=n))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, 0.15, size=n))
    return pd.DataFrame({"timestamp": ts, "open": opens, "high": highs, "low": lows, "close": closes})


def test_reformed_pipeline_runs_end_to_end_without_crashing():
    config = _toy_config(_always_on_sessions(), min_risk_atr_multiple=0.0)

    d1_df = _random_walk_candles(250, "1D", seed=1)
    h1_df = _random_walk_candles(2500, "1h", seed=2)
    m15_df = _random_walk_candles(4000, "15min", seed=3)

    d1_analysis = TimeframeAnalysis.from_candles(d1_df, "D1", config)
    h1_analysis = TimeframeAnalysis.from_candles(h1_df, "H1", config)
    m15_analysis = TimeframeAnalysis.from_candles(m15_df, "M15", config)

    generated = generate_reformed_signals("EURUSD", d1_analysis, h1_analysis, m15_analysis, config)

    assert isinstance(generated, list)

    for g in generated:
        s = g.signal
        assert s.symbol == "EURUSD"
        assert s.direction in ("long", "short")
        assert s.regime in ("trend_up", "trend_down", "range")
        assert s.dominant_reason_kind in ("liquidity_taken", "htf_zone", "key_level")
        assert s.dominant_reason_strength_percentile >= 80.0
        assert s.confirmation_score_percentile >= 70.0
        assert s.confirmation_followthrough is True
        assert CONVICTION_MIN <= s.conviction_multiplier <= CONVICTION_MAX
        assert abs(s.risk_pct_applied - BASE_RISK_PCT * s.conviction_multiplier) < 1e-9
        assert s.planned_risk_reward is not None and s.planned_risk_reward >= config.risk.min_risk_reward
        if s.direction == "long":
            assert s.original_sl < s.entry_reference_price < s.final_target
            if s.partial_target is not None:
                assert s.entry_reference_price < s.partial_target < s.final_target
        else:
            assert s.final_target < s.entry_reference_price < s.original_sl
            if s.partial_target is not None:
                assert s.final_target < s.partial_target < s.entry_reference_price
        assert 0 <= g.anchor_bar_idx < len(m15_df)
